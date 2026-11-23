#!/usr/bin/env python3
"""Strip `coleccion:<name>` tags from every Shopify product.

The previous pipeline added a `coleccion:<BrandFolder>` tag to every product
so the collection handle was visible to storefront filters. That experiment
is over -- strip those tags without touching anything else. Non-matching
tags are preserved as-is. Products whose only tag was the target tag end
up with an empty tag list.

The Shopify access token comes from the Shopify CLI session on the host, the
same one the image pipeline uses.

Usage
-----
    python shopify_strip_tags.py            # strip tags
    python shopify_strip_tags.py --dry-run  # report only, no writes
"""
import argparse
import sys

from shopify_api import graphql, load_token, raise_user_errors

TAG_PREFIX = "coleccion:"

PRODUCTS_QUERY = """
query Products($cursor: String) {
  products(first: 50, after: $cursor) {
    nodes {
      id
      title
      tags
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

PRODUCT_UPDATE_MUTATION = """
mutation UpdateProductTags($input: ProductInput!) {
  productUpdate(input: $input) {
    product { id }
    userErrors { field message }
  }
}
"""


def fetch_all_products(token):
    products = []
    cursor = None
    while True:
        data = graphql(token, PRODUCTS_QUERY, {"cursor": cursor})["products"]
        products.extend(data["nodes"])
        if not data["pageInfo"]["hasNextPage"]:
            return products
        cursor = data["pageInfo"]["endCursor"]


def plan_strips(products, prefix):
    plans = []
    for product in products:
        original = list(product.get("tags") or [])
        kept = [tag for tag in original if not tag.startswith(prefix)]
        if len(kept) == len(original):
            continue
        plans.append(
            {
                "id": product["id"],
                "title": product["title"],
                "before": original,
                "after": kept,
            }
        )
    return plans


def report_plan(plans):
    if not plans:
        print("Nothing to strip: no products carry the target tag.")
        return
    print(f"Plan: {len(plans)} product(s) need tag stripped")
    for plan in plans[:5]:
        before = ", ".join(plan["before"]) or "(empty)"
        after = ", ".join(plan["after"]) or "(empty)"
        print(f"  - {plan['title']}")
        print(f"      before: {before}")
        print(f"      after : {after}")
    if len(plans) > 5:
        print(f"  ... and {len(plans) - 5} more")


def strip_tags(token, product_id, new_tags):
    variables = {"input": {"id": product_id, "tags": new_tags}}
    payload = graphql(token, PRODUCT_UPDATE_MUTATION, variables)["productUpdate"]
    raise_user_errors("product tag update", payload["userErrors"])


def apply_strips(token, plans):
    total = len(plans)
    for index, plan in enumerate(plans, 1):
        strip_tags(token, plan["id"], plan["after"])
        print(f"  [{index}/{total}] {plan['title']} -> {len(plan['after'])} tag(s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove 'coleccion:' tags from every product"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing to Shopify",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = load_token()

    print("Fetching products...")
    products = fetch_all_products(token)
    print(f"  {len(products)} product(s) total")

    plans = plan_strips(products, TAG_PREFIX)
    report_plan(plans)

    if args.dry_run or not plans:
        return

    apply_strips(token, plans)
    print(f"Done: stripped coleccion tag from {len(plans)} product(s)")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, RuntimeError) as error:
        sys.exit(str(error))