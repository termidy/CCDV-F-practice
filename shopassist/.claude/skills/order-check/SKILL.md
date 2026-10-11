---
name: order-check
description: Check whether an order can still be returned. Use when someone asks about returning a specific order ID.
allowed-tools: Bash(python *)
---
Given an order ID:
1. Run `python -c "from mcp_server import lookup_order; print(lookup_order('<ORDER_ID>'))"` from the project folder.
2. If found, say how many days ago it was delivered and whether that is inside the 30-day return window.
3. If not found, say so and ask for the correct ID. Do not guess.
