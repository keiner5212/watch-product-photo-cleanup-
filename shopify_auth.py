#!/usr/bin/env python3
"""Upload local SKU images to matching Shopify product variants.

Prerequisites
------------
1. Install the Shopify CLI:  https://shopify.dev/docs/api/shopify-cli
2. Authenticate against the target store with the scopes this script
   needs (read+write products/inventory, read locations):

       shopify store auth --store axuzmh-jz.myshopify.com \\
           --scopes "read_products,write_products,read_inventory,write_inventory,read_locations"

3. The CLI stores the access token in a per-store config file
   (typically ~/.config/shopify/<store>.json on Linux). The token
   below is the one currently in the CLI config; re-running the
   `shopify store auth` command above prints + saves a new one --
   paste the new value into SHOPIFY_ACCESS_TOKEN below.

Expected image format
---------------------
- File naming:  iTrade-<SKU>.png   (e.g. iTrade-00507.png)
- Location:    ./output/<BrandFolder>/iTrade-<SKU>.png  (produced by
               process_images.py)
- Format:      PNG, transparent background (BGRA)
- Size:        up to 20 MB per file (Shopify hard limit, see
               MAX_IMAGE_BYTES); output is 1200x1200 so typical
               file is well under 1 MB
- Content:     square product shot, watch centred, brand text removed
               by the upstream pipeline

Usage
-----
    python shopify_auth.py            # upload + associate
    python shopify_auth.py --dry-run  # report matches only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

STORE = "axuzmh-jz.myshopify.com"
API_VERSION = "2026-07"
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = Path(os.environ.get("WATCH_OUTPUT_DIR", SCRIPT_DIR / "output")).resolve()
SHOPIFY_ACCESS_TOKEN = ""
GRAPHQL_URL = f"https://{STORE}/admin/api/{API_VERSION}/graphql.json"
STAGED_UPLOAD_HOST = "shopify-staged-uploads.storage.googleapis.com"
REQUEST_TIMEOUT_SECONDS = 60
MAX_REQUEST_ATTEMPTS = 4
MEDIA_READY_TIMEOUT_SECONDS = 120
MAX_RESPONSE_BYTES = 1_048_576
MAX_IMAGE_BYTES = 20_000_000
MAX_MUTATION_ITEMS = 250

PRODUCTS_QUERY = """
query Products($cursor: String) {
  products(first: 20, after: $cursor) {
    nodes {
      id
      title
      variants(first: 20) {
        nodes { id sku }
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
      nodes { id sku }
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

MEDIA_STATUS_QUERY = """
query MediaStatus($mediaId: ID!) {
  node(id: $mediaId) {
    ... on MediaImage { status }
  }
}
"""


def load_token():
    if not SHOPIFY_ACCESS_TOKEN:
        raise RuntimeError("SHOPIFY_ACCESS_TOKEN not set in shopify_auth.py")
    return SHOPIFY_ACCESS_TOKEN


def _graphql_error_message(errors):
    return "; ".join(error.get("message", "Unknown GraphQL error") for error in errors)


def _is_throttled(errors):
    return any(error.get("extensions", {}).get("code") == "THROTTLED" for error in errors)


def graphql(token, query, variables=None):
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": token,
    }

    for attempt in range(MAX_REQUEST_ATTEMPTS):
        request = Request(GRAPHQL_URL, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read(MAX_RESPONSE_BYTES))
        except HTTPError as error:
            if error.code == 429 or 500 <= error.code < 600:
                _retry_request(attempt, error.headers.get("Retry-After"))
                continue
            detail = error.read(MAX_RESPONSE_BYTES).decode(errors="replace")
            raise RuntimeError(f"Shopify HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            _retry_request(attempt)
            continue
        except json.JSONDecodeError as error:
            raise RuntimeError("Shopify returned invalid JSON") from error

        errors = result.get("errors", [])
        if errors and _is_throttled(errors):
            _retry_request(attempt)
            continue
        if errors:
            raise RuntimeError(f"Shopify GraphQL error: {_graphql_error_message(errors)}")
        if "data" not in result:
            raise RuntimeError("Shopify response missing data")
        return result["data"]

    raise RuntimeError("Shopify request failed after retries")


def _retry_request(attempt, retry_after=None):
    if attempt + 1 >= MAX_REQUEST_ATTEMPTS:
        return
    try:
        delay = float(retry_after) if retry_after else 2**attempt
    except ValueError:
        delay = 2**attempt
    time.sleep(delay)


def _raise_user_errors(operation, errors):
    if not errors:
        return
    details = "; ".join(error.get("message", "Unknown error") for error in errors)
    raise RuntimeError(f"Shopify {operation} failed: {details}")


def find_sku_images():
    if not OUTPUT_DIR.is_dir():
        raise RuntimeError(f"Image directory not found: {OUTPUT_DIR}")

    root = OUTPUT_DIR.resolve()
    images = {}
    for path in OUTPUT_DIR.rglob("iTrade-*.png"):
        if path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            continue
        if resolved.stat().st_size > MAX_IMAGE_BYTES:
            raise RuntimeError(f"Image exceeds Shopify 20 MB limit: {resolved}")
        sku = resolved.stem.removeprefix("iTrade-")
        if not sku:
            continue
        if sku in images:
            raise RuntimeError(
                f"Duplicate image for SKU {sku}: {images[sku]} and {resolved}"
            )
        images[sku] = resolved
    return images


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


def _media_filename(media):
    image = media.get("image") or {}
    url = image.get("url")
    if not url:
        return ""
    return unquote(Path(urlparse(url).path).name)


def find_existing_media(media_items, filename):
    for media in media_items:
        if media.get("alt") == filename or _media_filename(media) == filename:
            return media
    return None


def create_staged_upload(token, image_path):
    variables = {
        "input": [
            {
                "filename": image_path.name,
                "mimeType": "image/png",
                "httpMethod": "PUT",
                "resource": "IMAGE",
            }
        ]
    }
    payload = graphql(token, STAGED_UPLOAD_MUTATION, variables)["stagedUploadsCreate"]
    _raise_user_errors("staged upload", payload["userErrors"])
    targets = payload.get("stagedTargets") or []
    if not targets or not targets[0].get("url") or not targets[0].get("resourceUrl"):
        raise RuntimeError(f"Shopify returned no staged target for {image_path.name}")
    return targets[0]


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _staged_upload_headers(target):
    parsed = urlparse(target["url"])
    if parsed.scheme != "https" or parsed.hostname != STAGED_UPLOAD_HOST:
        raise RuntimeError(f"Shopify returned invalid staged upload host: {parsed.hostname}")

    headers = {}
    for parameter in target["parameters"]:
        name = parameter["name"]
        value = parameter["value"]
        normalized = name.lower()
        is_allowed = normalized in {"acl", "content-type", "content_type"}
        is_allowed = is_allowed or normalized.startswith(("x-goog-", "x-amz-"))
        if not is_allowed or "\r" in value or "\n" in value:
            raise RuntimeError(f"Shopify returned invalid staged upload header: {name}")
        headers[name] = value
    return headers


def upload_staged_file(target, image_path):
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise RuntimeError(f"Image exceeds Shopify 20 MB limit: {image_path}")

    request = Request(
        target["url"],
        data=image_path.read_bytes(),
        headers=_staged_upload_headers(target),
        method="PUT",
    )
    opener = build_opener(NoRedirectHandler())
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"Staged upload returned HTTP {response.status}")
    except HTTPError as error:
        detail = error.read(MAX_RESPONSE_BYTES).decode(errors="replace")
        raise RuntimeError(
            f"Failed to upload {image_path.name}: HTTP {error.code}: {detail}"
        ) from error
    except (OSError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Failed to upload {image_path.name}: {error}") from error


def attach_image_to_product(token, product_id, image_path):
    before_ids = {item["id"] for item in fetch_product_media(token, product_id)}
    target = create_staged_upload(token, image_path)
    upload_staged_file(target, image_path)
    variables = {
        "product": {"id": product_id},
        "media": [
            {
                "alt": image_path.name,
                "mediaContentType": "IMAGE",
                "originalSource": target["resourceUrl"],
            }
        ],
    }
    payload = graphql(token, PRODUCT_UPDATE_MUTATION, variables)["productUpdate"]
    _raise_user_errors("product update", payload["userErrors"])

    for media in fetch_product_media(token, product_id):
        if media["id"] not in before_ids and media.get("alt") == image_path.name:
            wait_for_media(token, media["id"], image_path.name)
            return media["id"]
    raise RuntimeError(f"Uploaded media not found on product for {image_path.name}")


def wait_for_media(token, media_id, filename):
    deadline = time.monotonic() + MEDIA_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        node = graphql(token, MEDIA_STATUS_QUERY, {"mediaId": media_id})["node"]
        status = node.get("status") if node else None
        if status == "READY":
            return
        if status == "FAILED":
            raise RuntimeError(f"Shopify failed to process {filename}")
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for Shopify to process {filename}")


def update_variant_media(token, product_id, assignments):
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
        _raise_user_errors("variant media update", payload["userErrors"])


def sync_product_images(token, products, sku_images):
    uploaded = 0
    associated = 0
    matched_products = 0

    for product in products:
        matches = [
            (variant, sku_images[variant["sku"]])
            for variant in product["variants"]
            if variant.get("sku") in sku_images
        ]
        if not matches:
            continue

        matched_products += 1
        media_items = fetch_product_media(token, product["id"])
        media_by_path = {}
        assignments = []
        for variant, image_path in matches:
            media_id = media_by_path.get(image_path)
            if not media_id:
                existing = find_existing_media(media_items, image_path.name)
                if existing:
                    media_id = existing["id"]
                else:
                    media_id = attach_image_to_product(token, product["id"], image_path)
                    media_items.append({"id": media_id, "alt": image_path.name})
                    uploaded += 1
                media_by_path[image_path] = media_id
            assignments.append((variant["id"], media_id))

        update_variant_media(token, product["id"], assignments)
        associated += len(assignments)
        print(f"Updated {product['title']}: {len(assignments)} variant image(s)")

    return matched_products, uploaded, associated


def report_image_matches(products, sku_images):
    matched_skus = set()
    matched_variants = 0
    missing_variants = 0
    variants_without_sku = 0

    for product in products:
        for variant in product["variants"]:
            sku = variant.get("sku")
            if not sku:
                variants_without_sku += 1
                print(f"[NO SKU] {product['title']}")
                continue
            image_path = sku_images.get(sku)
            if image_path:
                matched_skus.add(sku)
                matched_variants += 1
            else:
                missing_variants += 1
                print(f"[MISSING] SKU {sku} | iTrade-{sku}.png")

    unused_skus = sorted(set(sku_images) - matched_skus)
    for sku in unused_skus:
        print(f"[UNUSED] SKU {sku} | {sku_images[sku]}")

    print(
        f"Dry run: {matched_variants} variant(s) matched, "
        f"{missing_variants} missing image(s), "
        f"{variants_without_sku} variant(s) without SKU, "
        f"{len(unused_skus)} unused image(s)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match local SKU images with Shopify product variants"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report image matches without changing Shopify",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = load_token()
    sku_images = find_sku_images()
    products = fetch_products(token)
    print(f"Found {len(products)} product(s) and {len(sku_images)} SKU image(s)")
    if args.dry_run:
        report_image_matches(products, sku_images)
        return

    matched, uploaded, associated = sync_product_images(token, products, sku_images)
    print(
        f"Done: {matched} product(s) matched, {uploaded} image(s) uploaded, "
        f"{associated} variant(s) associated"
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, RuntimeError) as error:
        sys.exit(str(error))
