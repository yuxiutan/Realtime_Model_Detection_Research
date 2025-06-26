#!/bin/bash
curl -k -u admin:0000 -X GET "https://192.168.6.101:9200/wazuh-alerts-*/_search?pretty" -H 'Content-Type: application/json' -d'{
  "_source": ["agent.name", "agent.id", "agent.ip", "@timestamp", "rule.id", "rule.mitre.id", "full_log"],
  "query": {
    "bool": {
      "must": [
        {"range": {"@timestamp": {"gte": "now-5m", "lte": "now"}}},
        {"exists": {"field": "full_log"}}
      ]
    }
  },
  "sort": [{"@timestamp": {"order": "desc"}}],
  "size": 100
}' >> /home/youruser/Realtime_Transformer_Chain_Detection/data/new_attack_data.jsonl
