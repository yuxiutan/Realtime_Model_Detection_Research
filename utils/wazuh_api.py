# utils/wazuh_api.py
import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def save_wazuh_logs_to_file(
    file_path: str,
    url="https://192.168.6.228:9200",
    username="admin",
    password="0000",
    agent_name="modbus_device",
    time_range="now-5m",
    size=100
):
    headers = {'Content-Type': 'application/json'}
    endpoint = f"{url}/wazuh-alerts-*/_search?pretty"

    query = {
        "_source": [
            "agent.name", "agent.id", "agent.ip", "@timestamp",
            "rule.id", "rule.mitre.id", "rule.firedtimes", "full_log"
        ],
        "query": {
            "bool": {
                "must": [
                    {"match": {"agent.name": agent_name}},
                    {"range": {"@timestamp": {"gte": time_range, "lte": "now"}}},
                    {"exists": {"field": "full_log"}}
                ]
            }
        },
        "sort": [{"@timestamp": {"order": "desc"}}],
        "size": size
    }

    response = requests.get(
        endpoint,
        auth=(username, password),
        headers=headers,
        json=query,
        verify=False
    )

    if response.status_code == 200:
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(response.json(), f, indent=2)
            f.write("\n")
        print(f"Logs saved to {file_path}")
    else:
        raise RuntimeError(f"[{response.status_code}] Failed: {response.text}")
