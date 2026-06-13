"""Entry point helper with run instructions for the coffee inventory project."""


def main() -> None:
    print("Coffee inventory project ready.")
    print("Start API:   uvicorn server:app --reload")
    print("Run agent:   python agent.py")


if __name__ == "__main__":
    main()