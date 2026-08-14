import sys
import json

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
            method = msg.get("method")
            if method == "initialize":
                res = {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "dummy", "version": "1.0"}
                    }
                }
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                res = {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": "dummy_tool",
                                "description": "A dummy tool",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"]
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                args = msg.get("params", {}).get("arguments", {})
                res = {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "result": {
                        "content": [
                            {"type": "text", "text": f"Echo: {args.get('text', '')}"}
                        ]
                    }
                }
            else:
                res = {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "error": {"code": -32601, "message": "Method not found"}
                }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

if __name__ == "__main__":
    main()
