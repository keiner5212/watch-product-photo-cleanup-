#!/usr/bin/env python3
"""Assign `custom.sale_count` metafield to products that lack one.

Products in the "Mejor Vendidos" collection get a random value in
COLLECTION_RANGE; every other product gets a random value in DEFAULT_RANGE.
Products that already have sale_count are skipped (idempotent re-runs).

The Shopify access token is shared with shopify_auth.py - paste it there once
and both scripts pick it up.

Usage
-----
    python shopify_metafields.py            # assign missing values
    python shopify_metafields.py --dry-run  # report only, no writes
"""
import argparse
import random
import sys

from shopify_auth import _raise_user_errors, graphql, load_token

NAMESPACE = "custom"
KEY = "sale_count"
METAFIELD_TYPE = "number_integer"
COLLECTION_HANDLE = "mejor-vendidos"
COLLECTION_TITLE = "Mejor Vendidos"
DEFAULT_RANGE = (3, 10)
COLLECTION_RANGE = (10, 30)
PAGE_SIZE = 50

PRODUCTS_QUERY = """
query Products($cursor: String) {
  products(first: 50, after: $cursor) {
    nodes {
      id
      title
      saleCount: metafield(namespace: "custom", key: "sale_count") { value }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

COLLECTION_QUERY = """
query CollectionProducts($handle: String!, $cursor: String) {
  collectionByHandle(handle: $handle) {
    id
    title
    products(first: 50, after: $cursor) {
      nodes { id }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

METAFIELDS_SET_MUTATION = """
mutation MetafieldsSet($metafields: [MetafieldsSetInput!]!) {
  metafieldsSet(metafields: $metafields) {
    metafields { id }
    userErrors { field message }
  }
}
"""


def fetch_collection_product_ids(token, handle):
    cursor = None
    ids = set()
    while True:
        data = graphql(token, COLLECTION_QUERY, {"handle": handle, "cursor": cursor})
        collection = data.get("collectionByHandle")
        if not collection:
            raise RuntimeError(f"Collection not found: {handle}")
        connection = collection["products"]
        ids.update(node["id"] for node in connection["nodes"])
        if not connection["pageInfo"]["hasNextPage"]:
            return ids
        cursor = connection["pageInfo"]["endCursor"]


def fetch_all_products(token):
    products = []
    cursor = None
    while True:
        data = graphql(token, PRODUCTS_QUERY, {"cursor": cursor})["products"]
        products.extend(data["nodes"])
        if not data["pageInfo"]["hasNextPage"]:
            return products
        cursor = data["pageInfo"]["endCursor"]


def assign_sale_count(token, product_id, value):
    variables = {
        "metafields": [
            {
                "ownerId": product_id,
                "namespace": NAMESPACE,
                "key": KEY,
                "type": METAFIELD_TYPE,
                "value": str(value),
            }
        ]
    }
    payload = graphql(token, METAFIELDS_SET_MUTATION, variables)["metafieldsSet"]
    _raise_user_errors("metafield set", payload["userErrors"])


def plan_assignments(products, collection_ids):
    plans = []
    for product in products:
        existing = product.get("saleCount")
        if existing and existing.get("value") not in (None, ""):
            continue
        in_collection = product["id"] in collection_ids
        low, high = COLLECTION_RANGE if in_collection else DEFAULT_RANGE
        plans.append(
            {
                "id": product["id"],
                "title": product["title"],
                "in_collection": in_collection,
                "value": random.randint(low, high),
            }
        )
    return plans


def report_plan(plans):
    if not plans:
        print("Nothing to assign: every product already has sale_count.")
        return
    in_coll = sum(1 for p in plans if p["in_collection"])
    print(f"Plan: {len(plans)} product(s) need sale_count")
    print(f"  in '{COLLECTION_TITLE}': {in_coll}")
    print(f"  others: {len(plans) - in_coll}")
    for plan in plans[:5]:
        where = "in coll" if plan["in_collection"] else "default"
        print(f"  - {plan['title']}: {plan['value']} ({where})")
    if len(plans) > 5:
        print(f"  ... and {len(plans) - 5} more")


def apply_assignments(token, plans):
    total = len(plans)
    for index, plan in enumerate(plans, 1):
        assign_sale_count(token, plan["id"], plan["value"])
        print(f"  [{index}/{total}] {plan['title']} -> {plan['value']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign sale_count metafield to products that lack one"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be assigned without writing to Shopify",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = load_token()

    print(f"Fetching collection '{COLLECTION_HANDLE}'...")
    collection_ids = fetch_collection_product_ids(token, COLLECTION_HANDLE)
    print(f"  {len(collection_ids)} product(s) in collection")

    print("Fetching products...")
    products = fetch_all_products(token)
    print(f"  {len(products)} product(s) total")

    plans = plan_assignments(products, collection_ids)
    report_plan(plans)

    if args.dry_run or not plans:
        return

    apply_assignments(token, plans)
    print(f"Done: assigned sale_count to {len(plans)} product(s)")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, RuntimeError) as error:
        sys.exit(str(error))
