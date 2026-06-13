import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("INVENTORY_API_BASE_URL", "http://127.0.0.1:8000")
LOG_FILE = Path("conversation_log.csv")
LOG_HEADERS = ["actor", "message", "tool_call", "timestamp"]
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "List all inventory products.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_product",
            "description": "Register a new product in inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "stock": {"type": "integer", "minimum": 0},
                    "min_stock": {"type": "integer", "minimum": 0},
                    "unit": {"type": "string"},
                },
                "required": ["name", "stock", "min_stock"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": "Update stock quantity for an existing product by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "stock": {"type": "integer", "minimum": 0},
                },
                "required": ["product_id", "stock"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "low_stock_alerts",
            "description": "Get products that are at or below min_stock.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_log_file() -> None:
    if LOG_FILE.exists():
        return
    with LOG_FILE.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_HEADERS)
        writer.writeheader()


def append_log(actor: str, message: str, tool_call: str = "") -> None:
    ensure_log_file()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_HEADERS)
        writer.writerow(
            {
                "actor": actor,
                "message": message,
                "tool_call": tool_call,
                "timestamp": now_iso(),
            }
        )


def tool_list_products() -> List[Dict[str, Any]]:
    response = requests.get(f"{API_BASE_URL}/products", timeout=20)
    response.raise_for_status()
    return response.json()


def tool_register_product(name: str, stock: int, min_stock: int, unit: str = "units") -> Dict[str, Any]:
    payload = {"name": name, "stock": stock, "min_stock": min_stock, "unit": unit}
    response = requests.post(f"{API_BASE_URL}/products", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def tool_update_stock(product_id: str, stock: int) -> Dict[str, Any]:
    response = requests.patch(
        f"{API_BASE_URL}/products/{product_id}/stock",
        json={"stock": stock},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def tool_low_stock_alerts() -> List[Dict[str, Any]]:
    response = requests.get(f"{API_BASE_URL}/alerts/low-stock", timeout=20)
    response.raise_for_status()
    return response.json()


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    if tool_name == "list_products":
        return tool_list_products()
    if tool_name == "register_product":
        return tool_register_product(
            name=arguments["name"],
            stock=arguments["stock"],
            min_stock=arguments["min_stock"],
            unit=arguments.get("unit", "units"),
        )
    if tool_name == "update_stock":
        return tool_update_stock(product_id=arguments["product_id"], stock=arguments["stock"])
    if tool_name == "low_stock_alerts":
        return tool_low_stock_alerts()
    raise ValueError(f"Unknown tool: {tool_name}")


def run_agent_loop() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a coffee inventory assistant for Carla. "
                "Use tools to answer inventory questions and perform updates. "
                "When enough information is available, provide a concise final answer."
            ),
        }
    ]

    print("Inventory agent started. Type 'exit' to quit.")
    while True:
        user_message = input("Carla> ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        append_log(actor="user", message=user_message, tool_call="")
        messages.append({"role": "user", "content": user_message})

        for _ in range(6):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            choice = response.choices[0].message
            tool_calls = choice.tool_calls or []

            if not tool_calls:
                final_message = choice.content or "I could not generate a response."
                messages.append({"role": "assistant", "content": final_message})
                append_log(actor="agent", message=final_message, tool_call="")
                print(f"Agent> {final_message}")
                break

            messages.append(choice.model_dump())
            assistant_intent = choice.content or "Calling a tool."
            first_tool_name = tool_calls[0].function.name if tool_calls else ""
            append_log(actor="agent", message=assistant_intent, tool_call=first_tool_name)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")
                try:
                    result = call_tool(tool_name=tool_name, arguments=arguments)
                except Exception as error:
                    result = {"error": str(error)}

                result_text = json.dumps(result, ensure_ascii=True)
                append_log(actor="tool", message=result_text, tool_call=tool_name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )
        else:
            timeout_message = "I hit the loop limit before reaching a final answer."
            append_log(actor="agent", message=timeout_message, tool_call="")
            print(f"Agent> {timeout_message}")


if __name__ == "__main__":
    run_agent_loop()
