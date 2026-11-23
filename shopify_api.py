#!/usr/bin/env python3
"""Shopify Admin GraphQL client used by the watch image pipeline.

The admin API token is not stored in this repo. It is read from the config
file the Shopify CLI writes on the host:

    shopify store auth --store axuzmh-jz.myshopify.com \
        --scopes "read_products,write_products,read_inventory,write_inventory,read_locations"

The CLI saves that session in ~/.config/shopify-cli-store-nodejs/config.json,
which the compose file mounts read-only at /data/shopify-cli.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

STORE = os.environ.get("SHOPIFY_STORE", "axuzmh-jz.myshopify.com")
API_VERSION = os.environ.get("SHOPIFY_API_VERSION", "2026-07")
CLI_CONFIG_DIR = Path(
    os.environ.get("SHOPIFY_CLI_CONFIG_DIR", "/data/shopify-cli")
)
OUTPUT_DIR = Path(
    os.environ.get("WATCH_OUTPUT_DIR", Path(__file__).parent / "output")
).resolve()
GRAPHQL_URL = f"https://{STORE}/admin/api/{API_VERSION}/graphql.json"
STAGED_UPLOAD_HOST = "shopify-staged-uploads.storage.googleapis.com"

REQUIRED_SCOPES = ("write_products",)
SKU_PATTERN = re.compile(r"iTrade-(\d+)", re.IGNORECASE)
STORE_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*\.myshopify\.com")
API_VERSION_PATTERN = re.compile(r"\d{4}-\d{2}")
HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
MAX_ERROR_DETAIL = 500

REQUEST_TIMEOUT_SECONDS = 60
MAX_REQUEST_ATTEMPTS = 4
MEDIA_READY_TIMEOUT_SECONDS = 180
MAX_RESPONSE_BYTES = 8_388_608
MAX_IMAGE_BYTES = 20_000_000
MAX_MUTATION_ITEMS = 250
MAX_VARIANT_MEDIA_PAIRS = 100
THROTTLE_HEADROOM = 50


class ShopifyError(RuntimeError):
    """Shopify refused or could not complete a request."""


PRODUCTS_QUERY = """
query Products($cursor: String) {
  products(first: 20, after: $cursor) {
    nodes {
      id
      title
      variants(first: 20) {
        nodes {
          id
          sku
          media(first: 10) { nodes { id } }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

PRODUCT_VARIANTS_QUERY = """
query ProductVariants($productId: ID!, $cursor: String) {
  product(id: $productId) {
    variants(first: 100, after: $cursor) {
      nodes {
        id
        sku
        media(first: 10) { nodes { id } }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

PRODUCT_MEDIA_QUERY = """
query ProductMedia($productId: ID!, $cursor: String) {
  product(id: $productId) {
    media(first: 100, after: $cursor) {
      nodes {
        id
        alt
        status
        ... on MediaImage { image { url } }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

STAGED_UPLOAD_MUTATION = """
mutation StagedUpload($input: [StagedUploadInput!]!) {
  stagedUploadsCreate(input: $input) {
    stagedTargets {
      url
      resourceUrl
      parameters { name value }
    }
    userErrors { field message }
  }
}
"""

PRODUCT_UPDATE_MUTATION = """
mutation AddProductMedia($product: ProductUpdateInput!, $media: [CreateMediaInput!]) {
  productUpdate(product: $product, media: $media) {
    product { id }
    userErrors { field message }
  }
}
"""

VARIANTS_UPDATE_MUTATION = """
mutation UpdateVariantMedia($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
  productVariantsBulkUpdate(productId: $productId, variants: $variants) {
    productVariants { id }
    userErrors { field message }
  }
}
"""

VARIANT_DETACH_MEDIA_MUTATION = """
mutation DetachVariantMedia($productId: ID!, $variantMedia: [ProductVariantDetachMediaInput!]!) {
  productVariantDetachMedia(productId: $productId, variantMedia: $variantMedia) {
    productVariants { id }
    userErrors { field message }
  }
}
"""

PRODUCT_DELETE_MEDIA_MUTATION = """
mutation DeleteProductMedia($productId: ID!, $mediaIds: [ID!]!) {
  productDeleteMedia(productId: $productId, mediaIds: $mediaIds) {
    deletedMediaIds
    mediaUserErrors { field message }
    userErrors { field message }
  }
}
"""

MEDIA_STATUS_QUERY = """
query MediaStatus($mediaId: ID!) {
  node(id: $mediaId) {
    ... on MediaImage { status }
  }
}
"""


def _iter_sessions(node):
    if isinstance(node, dict):
        if "accessToken" in node and "store" in node:
            yield node
        for value in node.values():
            yield from _iter_sessions(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_sessions(item)


def _session_expiry(session):
    raw = session.get("expiresAt")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def find_store_session(store=STORE, config_dir=CLI_CONFIG_DIR):
    """Return the CLI session for `store` with the latest expiry."""
    if not config_dir.is_dir():
        return None
    sessions = []
    for path in sorted(config_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for session in _iter_sessions(data):
            if session.get("store") == store:
                sessions.append(session)
    if not sessions:
        return None
    return max(sessions, key=lambda s: _session_expiry(s) or datetime.min.replace(tzinfo=timezone.utc))


def load_token():
    """Return a live admin API token for STORE, or explain how to get one."""
    if not STORE_PATTERN.fullmatch(STORE):
        raise ShopifyError(f"invalid store domain: {STORE!r}")
    if not API_VERSION_PATTERN.fullmatch(API_VERSION):
        raise ShopifyError(f"invalid API version: {API_VERSION!r}")
    session = find_store_session()
    if session is None:
        raise ShopifyError(
            f"no Shopify CLI session for {STORE} under {CLI_CONFIG_DIR}. "
            "Run on the host:\n"
            f"  shopify store auth --store {STORE} --scopes "
            '"read_products,write_products,read_inventory,write_inventory,read_locations"'
        )

    token = session.get("accessToken")
    if not token:
        raise ShopifyError(f"Shopify CLI session for {STORE} carries no access token")

    expiry = _session_expiry(session)
    if expiry is not None and expiry <= datetime.now(timezone.utc):
        raise ShopifyError(
            f"Shopify token for {STORE} expired at {expiry.isoformat()}. "
            f"Run on the host:\n  shopify store auth --store {STORE} --scopes "
            '"read_products,write_products,read_inventory,write_inventory,read_locations"'
        )

    scopes = set(session.get("scopes") or [])
    missing = [scope for scope in REQUIRED_SCOPES if scope not in scopes]
    if missing:
        raise ShopifyError(
            f"Shopify token for {STORE} is missing scope(s) {', '.join(missing)}. "
            "Re-auth with the full scope list."
        )
    return token


def _graphql_error_message(errors):
    return "; ".join(error.get("message", "Unknown GraphQL error") for error in errors)


def _is_throttled(errors):
    return any(error.get("extensions", {}).get("code") == "THROTTLED" for error in errors)


def _wait_for_quota(result):
    """Sleep when the GraphQL cost quota is nearly exhausted."""
    throttle = result.get("extensions", {}).get("cost", {}).get("throttleStatus")
    if not throttle:
        return
    available = throttle.get("currentlyAvailable", 100)
    if available >= THROTTLE_HEADROOM:
        return
    restore_rate = throttle.get("restoreRate", 1000)
    delay = (THROTTLE_HEADROOM - available) / max(restore_rate, 1)
    time.sleep(min(delay, 5.0))


def graphql(token, query, variables=None):
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": token,
    }

    for attempt in range(MAX_REQUEST_ATTEMPTS):
        request = Request(GRAPHQL_URL, data=payload, headers=headers, method="POST")
        try:
            with _opener().open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                if urlparse(response.geturl()).hostname != STORE:
                    raise ShopifyError("Shopify API redirected to an unexpected host")
                result = json.loads(response.read(MAX_RESPONSE_BYTES))
        except HTTPError as error:
            if error.code == 429 or 500 <= error.code < 600:
                _retry_request(attempt, error.headers.get("Retry-After"))
                continue
            detail = error.read(MAX_RESPONSE_BYTES).decode(errors="replace")
            raise ShopifyError(
                f"Shopify HTTP {error.code}: {detail[:MAX_ERROR_DETAIL]}"
            ) from error
        except (URLError, TimeoutError) as error:
            _retry_request(attempt)
            continue
        except json.JSONDecodeError as error:
            raise ShopifyError("Shopify returned invalid JSON") from error

        errors = result.get("errors", [])
        if errors and _is_throttled(errors):
            _retry_request(attempt)
            continue
        if errors:
            raise ShopifyError(f"Shopify GraphQL error: {_graphql_error_message(errors)}")
        if "data" not in result:
            raise ShopifyError("Shopify response missing data")
        _wait_for_quota(result)
        return result["data"]

    raise ShopifyError("Shopify request failed after retries")


def _retry_request(attempt, retry_after=None):
    if attempt + 1 >= MAX_REQUEST_ATTEMPTS:
        return
    try:
        delay = float(retry_after) if retry_after else 2**attempt
    except ValueError:
        delay = 2**attempt
    time.sleep(delay)


def raise_user_errors(operation, errors):
    if not errors:
        return
    details = "; ".join(error.get("message", "Unknown error") for error in errors)
    raise ShopifyError(f"Shopify {operation} failed: {details}")


def fetch_products(token):
    products = []
    cursor = None
    while True:
        connection = graphql(token, PRODUCTS_QUERY, {"cursor": cursor})["products"]
        for product in connection["nodes"]:
            product["variants"] = _fetch_all_variants(token, product)
            products.append(product)
        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            return products
        cursor = page_info["endCursor"]


def _fetch_all_variants(token, product):
    connection = product["variants"]
    variants = connection["nodes"]
    cursor = connection["pageInfo"]["endCursor"]
    while connection["pageInfo"]["hasNextPage"]:
        data = graphql(
            token,
            PRODUCT_VARIANTS_QUERY,
            {"productId": product["id"], "cursor": cursor},
        )
        connection = data["product"]["variants"]
        variants.extend(connection["nodes"])
        cursor = connection["pageInfo"]["endCursor"]
    return variants


def fetch_product_media(token, product_id):
    media = []
    cursor = None
    while True:
        data = graphql(
            token,
            PRODUCT_MEDIA_QUERY,
            {"productId": product_id, "cursor": cursor},
        )
        connection = data["product"]["media"]
        media.extend(connection["nodes"])
        if not connection["pageInfo"]["hasNextPage"]:
            return media
        cursor = connection["pageInfo"]["endCursor"]


def media_sku(media):
    """Return the SKU a media item belongs to, or None when it is not ours."""
    alt = media.get("alt") or ""
    match = SKU_PATTERN.search(alt)
    if match:
        return match.group(1)
    url = (media.get("image") or {}).get("url") or ""
    filename = unquote(Path(urlparse(url).path).name)
    match = SKU_PATTERN.search(filename)
    return match.group(1) if match else None


def create_staged_upload(token, image_path, mime_type="image/webp"):
    variables = {
        "input": [
            {
                "filename": image_path.name,
                "mimeType": mime_type,
                "httpMethod": "PUT",
                "resource": "IMAGE",
            }
        ]
    }
    payload = graphql(token, STAGED_UPLOAD_MUTATION, variables)["stagedUploadsCreate"]
    raise_user_errors("staged upload", payload["userErrors"])
    targets = payload.get("stagedTargets") or []
    if not targets or not targets[0].get("url") or not targets[0].get("resourceUrl"):
        raise ShopifyError(f"Shopify returned no staged target for {image_path.name}")
    return targets[0]


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, msg, headers, fp)


def _opener():
    return build_opener(NoRedirectHandler())


def _staged_upload_headers(target):
    parsed = urlparse(target["url"])
    if parsed.scheme != "https" or parsed.hostname != STAGED_UPLOAD_HOST:
        raise ShopifyError(f"Shopify returned invalid staged upload host: {parsed.hostname}")
    if parsed.port not in (None, 443):
        raise ShopifyError(f"Shopify returned invalid staged upload port: {parsed.port}")

    headers = {}
    for parameter in target["parameters"]:
        name = parameter["name"]
        value = parameter["value"]
        normalized = name.lower()
        is_allowed = normalized in {"acl", "content-type", "content_type"}
        is_allowed = is_allowed or normalized.startswith(("x-goog-", "x-amz-"))
        if not is_allowed or "\r" in value or "\n" in value:
            raise ShopifyError(f"Shopify returned invalid staged upload header: {name}")
        if not HEADER_NAME_PATTERN.fullmatch(name):
            raise ShopifyError(f"Shopify returned invalid staged upload header name: {name!r}")
        headers[name] = value
    return headers


def _read_upload_bytes(image_path):
    """Read the image without following links out of the output folder."""
    resolved = image_path.resolve()
    if not resolved.is_relative_to(OUTPUT_DIR.resolve()):
        raise ShopifyError(f"refusing to upload a file outside {OUTPUT_DIR}: {image_path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(resolved, flags)
    with os.fdopen(handle, "rb") as stream:
        data = stream.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ShopifyError(f"Image exceeds Shopify 20 MB limit: {image_path}")
    if not data.startswith(b"RIFF") or data[8:12] != b"WEBP":
        raise ShopifyError(f"not a WEBP file: {image_path}")
    return data


def upload_staged_file(target, image_path):
    request = Request(
        target["url"],
        data=_read_upload_bytes(image_path),
        headers=_staged_upload_headers(target),
        method="PUT",
    )
    try:
        with _opener().open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status < 200 or response.status >= 300:
                raise ShopifyError(f"Staged upload returned HTTP {response.status}")
            if urlparse(response.geturl()).hostname != STAGED_UPLOAD_HOST:
                raise ShopifyError("Staged upload was redirected to an unexpected host")
    except HTTPError as error:
        detail = error.read(MAX_RESPONSE_BYTES).decode(errors="replace")
        raise ShopifyError(
            f"Failed to upload {image_path.name}: HTTP {error.code}: {detail[:MAX_ERROR_DETAIL]}"
        ) from error
    except (OSError, URLError, TimeoutError) as error:
        raise ShopifyError(f"Failed to upload {image_path.name}: {error}") from error


def attach_image_to_product(token, product_id, image_path, alt):
    """Upload one image to a product and return the new media id."""
    before_ids = {item["id"] for item in fetch_product_media(token, product_id)}
    target = create_staged_upload(token, image_path)
    upload_staged_file(target, image_path)
    variables = {
        "product": {"id": product_id},
        "media": [
            {
                "alt": alt,
                "mediaContentType": "IMAGE",
                "originalSource": target["resourceUrl"],
            }
        ],
    }
    payload = graphql(token, PRODUCT_UPDATE_MUTATION, variables)["productUpdate"]
    raise_user_errors("product update", payload["userErrors"])

    for media in fetch_product_media(token, product_id):
        if media["id"] not in before_ids and media.get("alt") == alt:
            wait_for_media(token, media["id"], image_path.name)
            return media["id"]
    raise ShopifyError(f"Uploaded media not found on product for {image_path.name}")


def wait_for_media(token, media_id, filename):
    deadline = time.monotonic() + MEDIA_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        node = graphql(token, MEDIA_STATUS_QUERY, {"mediaId": media_id})["node"]
        status = node.get("status") if node else None
        if status == "READY":
            return
        if status == "FAILED":
            raise ShopifyError(f"Shopify failed to process {filename}")
        time.sleep(2)
    raise ShopifyError(f"Timed out waiting for Shopify to process {filename}")


def set_variant_media(token, product_id, assignments):
    """Point each variant at its media image."""
    for index in range(0, len(assignments), MAX_MUTATION_ITEMS):
        batch = assignments[index : index + MAX_MUTATION_ITEMS]
        variables = {
            "productId": product_id,
            "variants": [
                {"id": variant_id, "mediaId": media_id}
                for variant_id, media_id in batch
            ],
        }
        payload = graphql(token, VARIANTS_UPDATE_MUTATION, variables)[
            "productVariantsBulkUpdate"
        ]
        raise_user_errors("variant media update", payload["userErrors"])


def detach_variant_media(token, product_id, pairs):
    """Remove one media from one variant. The API takes a single media per entry."""
    for index in range(0, len(pairs), MAX_VARIANT_MEDIA_PAIRS):
        batch = pairs[index : index + MAX_VARIANT_MEDIA_PAIRS]
        variables = {
            "productId": product_id,
            "variantMedia": [
                {"variantId": variant_id, "mediaIds": [media_id]}
                for variant_id, media_id in batch
            ],
        }
        payload = graphql(token, VARIANT_DETACH_MEDIA_MUTATION, variables)[
            "productVariantDetachMedia"
        ]
        raise_user_errors("variant media detach", payload["userErrors"])


def delete_product_media(token, product_id, media_ids):
    """Delete media from a product."""
    for index in range(0, len(media_ids), MAX_MUTATION_ITEMS):
        batch = media_ids[index : index + MAX_MUTATION_ITEMS]
        payload = graphql(
            token,
            PRODUCT_DELETE_MEDIA_MUTATION,
            {"productId": product_id, "mediaIds": batch},
        )["productDeleteMedia"]
        raise_user_errors(
            "media delete",
            payload.get("mediaUserErrors") or payload.get("userErrors"),
        )


if __name__ == "__main__":
    try:
        load_token()
        print(f"Shopify token loaded for {STORE}")
    except ShopifyError as error:
        sys.exit(str(error))
