import json
import numpy as np
import pandas as pd
import joblib
import time
import logging
import os
import re
import requests
from datetime import datetime, timedelta
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import warnings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

warnings.filterwarnings('ignore')

class DiscordNotifier:
    """Discord webhook notification handler"""
    def __init__(self, webhook_url=None, enable_discord=True):
        self.webhook_url = webhook_url
        self.enable_discord = enable_discord and webhook_url is not None
        self.logger = logging.getLogger(__name__ + '.discord')
    
        if self.enable_discord:
            self.logger.info("Discord notifications enabled")
        else:
            self.logger.warning("Discord notifications disabled (no webhook URL provided)")

        self.frequency_threshold = {
            'time_window_minutes': 5,
            'min_occurrences': 3,
            'min_records': 3
        }
    
    def send_discord_alert(self, alert_data):
        """Send alert to Discord webhook"""
        if not self.enable_discord:
            return False
        
        try:
            # Determine alert color based on layer and severity
            color = self._get_alert_color(alert_data)
            
            # Create Discord embed
            embed = {
                "title": f"{alert_data.get('alert_type', 'SECURITY ALERT')}",
                "description": self._format_alert_description(alert_data),
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "fields": self._create_alert_fields(alert_data),
                "footer": {
                    "text": f"Security Detection System - Layer {alert_data.get('detection_layer', 'Unknown')}"
                }
            }
            
            # Prepare webhook payload
            payload = {
                "content": self._get_mention_content(alert_data),
                "embeds": [embed],
                "username": "Security Detection Bot",
                "avatar_url": "https://cdn.discordapp.com/attachments/example/security_bot_avatar.png"
            }
            
            # Send to Discord
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 204:
                self.logger.info("Discord alert sent successfully")
                return True
            else:
                self.logger.error(f"Discord webhook failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Discord webhook request failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending Discord alert: {e}")
            return False
    
    def _get_alert_color(self, alert_data):
        """Get Discord embed color based on alert severity"""
        layer = alert_data.get('detection_layer', 1)
        confidence = alert_data.get('confidence', 0.0)
    
        if layer == 1:
            return 16711680
        elif confidence >= 0.8:
            return 16711680 
        elif confidence >= 0.6:
            return 16753920
        else:
            return 16776960
    
    def _format_alert_description(self, alert_data):
        """Format alert description for Discord embed"""
        layer = alert_data.get('detection_layer', 1)
        scenario = alert_data.get('scenario', '')
        confidence = alert_data.get('confidence', 0.0)
    
        if layer == 1:
            return f"**Detection Layer**: Layer 1 (Frequency Detection)\n**Threat Level**: CRITICAL\n**Technique ID**: {alert_data.get('technique_id', 'T0831')}"
        else:
            return f"**Detection Layer**: Layer 2 (LSTM Attack Chain Analysis)\n**Threat Type**: {scenario}\n**Confidence**: {confidence:.3f}\n**Technique ID**: {alert_data.get('technique_id', 'Unknown')}"
    
    def _create_alert_fields(self, alert_data):
        """Create alert fields"""
        raw_log = alert_data.get('raw_log', {})
        
        fields = []
        
        # Agent information
        if raw_log.get('agent.ip') or raw_log.get('agent.name'):
            fields.append({
                "name": "Agent Information",
                "value": f"**IP**: {raw_log.get('agent.ip', 'Unknown')}\n**Name**: {raw_log.get('agent.name', 'Unknown')}\n**ID**: {raw_log.get('agent.id', 'Unknown')}",
                "inline": True
            })
        
        # Rule information
        if raw_log.get('rule.id'):
            fields.append({
                "name": "Rule Information",
                "value": f"**Rule ID**: {raw_log.get('rule.id', 'Unknown')}\n**MITRE ID**: {raw_log.get('rule.mitre.id', 'Unknown')}",
                "inline": True
            })
        
        # Timestamp
        if raw_log.get('@timestamp'):
            fields.append({
                "name": "Timestamp",
                "value": f"{raw_log.get('@timestamp', 'Unknown')}",
                "inline": False
            })
        
        # Log details (truncate long logs)
        full_log = raw_log.get('full_log', '')
        if full_log:
            log_preview = full_log[:500] + "..." if len(full_log) > 500 else full_log
            fields.append({
                "name": "Log Details",
                "value": f"```{log_preview}```",
                "inline": False
            })
        
        return fields
    
    def _get_mention_content(self, alert_data):
        """Get mention content (configurable @everyone or specific users)"""
        layer = alert_data.get('detection_layer', 1)
        scenario = alert_data.get('scenario', '')
        
        # For critical threats, can @everyone
        critical_threats = ['Frequency Request', 'IT_Ransomware', 'IT_LSASS_Dump']
        
        if layer == 1 or scenario in critical_threats:
            return " **CRITICAL SECURITY ALERT** "  # Can change to "@everyone" if needed
        else:
            return " **Security Alert** "

class LogFileHandler(FileSystemEventHandler):
    """File monitoring handler"""
    def __init__(self, detector):
        self.detector = detector
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('wazuh_log.log'):
            self.detector.process_new_logs()

class IntegratedSecurityDetector:
    def __init__(self, 
                 log_file_path="/home/danish/Realtime_LSTM_Chain_Detection/data/wazuh_log.log",
                 model_path="model/lstm_attack_chain_model.keras",
                 encoders_path="encoders.pkl",
                 classes_path="classes",
                 threshold=0.7,
                 sequence_length=10,
                 detection_log="integrated_security_detection.log",
                 alert_cooldown=300,
                 discord_webhook_url=None,
                 enable_discord_alerts=True):
        
        self.frequency_buffer = deque(maxlen=100)
        self.frequency_threshold = {
            'time_window_minutes': 5,
            'firedtimes_diff_threshold': 2,
            'min_records': 2
        }
        self.log_file_path = log_file_path
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.classes_path = classes_path
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.detection_log = detection_log
        self.alert_cooldown = alert_cooldown
        
        self.discord_notifier = DiscordNotifier(
                webhook_url=discord_webhook_url,
                enable_discord=enable_discord_alerts
        )
        
        # Initialize logging
        self.setup_logging()
        
        # Load model and encoders (for second layer)
        self.load_model_and_encoders()
        
        # Initialize sequence buffer (for second layer)
        self.sequence_buffer = deque(maxlen=sequence_length)
        
        # Record processed file position
        self.last_position = 0
        self.processed_logs = deque(maxlen=50)
        
        # MITRE ID sequence tracking (for second layer special rules)
        self.mitre_sequence = deque(maxlen=20)
        
        # Alert status management
        self.active_alerts = {}  # Store active alerts {attack_type: last_alert_time}
        self.layer1_active_alerts = {}  # Store Layer 1 alerts {alert_type: last_alert_time}
        self.layer1_alert_cooldown = alert_cooldown  # Can be same or different from Layer 2
        self.detected_attack_chains = set()
        self.last_special_rule_trigger = {}
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'layer1_detections': 0,  # Frequency detections
            'layer2_detections': 0,  # LSTM model detections
            'layer2_special_rules': 0,  # Special rule detections
            'total_alerts': 0
        }
        
        # Ensure log file exists
        if not os.path.exists(self.log_file_path):
            open(self.log_file_path, 'a').close()
            
        self.logger.info("Integrated Two-Layer Security Detector initialized successfully")
        self.logger.info("Layer 1: Frequency Critical Detection")
        self.logger.info("Layer 2: LSTM Attack Chain Analysis")
        self.logger.info(f"Alert cooldown period: {self.alert_cooldown} seconds")
        self.logger.info("Discord alert functionality initialized")

    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.detection_log),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_encoders(self):
        """Load trained model and label encoders for second layer"""
        try:
            # Load LSTM model
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.logger.info(f"LSTM model loaded from {self.model_path}")
            else:
                self.logger.warning(f"LSTM model not found at {self.model_path}. Layer 2 will be limited.")
                self.model = None
            
            # Load label encoders
            if os.path.exists(self.encoders_path):
                self.encoders = joblib.load(self.encoders_path)
                self.logger.info(f"Encoders loaded from {self.encoders_path}")
                
                # Load class labels
                if 'attack_chain' in self.encoders:
                    self.attack_chain_classes = self.encoders['attack_chain'].classes_
                else:
                    self.attack_chain_classes = []
            else:
                self.logger.warning(f"Encoders not found at {self.encoders_path}. Layer 2 will be limited.")
                self.encoders = {}
                self.attack_chain_classes = []
            
            # Load feature classes
            self.load_feature_classes()
            
        except Exception as e:
            self.logger.error(f"Error loading model or encoders: {e}")
            self.model = None
            self.encoders = {}
            self.attack_chain_classes = []
    
    def load_feature_classes(self):
        """Load possible value ranges for each feature"""
        self.feature_classes = {}
        
        try:
            if not os.path.exists(self.classes_path):
                self.logger.warning(f"Classes directory not found: {self.classes_path}")
                return
                
            # Numeric features
            numeric_features = ['agent_ip', 'agent_id', 'rule_id']
            for feature in numeric_features:
                class_file = os.path.join(self.classes_path, f"{feature}_classes.npy")
                if os.path.exists(class_file):
                    self.feature_classes[feature] = np.load(class_file, allow_pickle=True)
            
            # Encoded features
            encoded_features = ['agent_name', 'eventdata_image', 'mitre_id']
            for feature in encoded_features:
                class_file = os.path.join(self.classes_path, f"{feature}_classes.npy")
                if os.path.exists(class_file):
                    self.feature_classes[feature] = np.load(class_file, allow_pickle=True)
                    
            self.logger.info("Feature classes loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load feature classes: {e}")
            self.feature_classes = {}
    
    def extract_json_objects(self, text):
        """Extract JSON objects from text, handling nested cases"""
        result = []
        pos = 0
        length = len(text)
        
        while pos < length:
            # Find next { character
            start = text.find('{', pos)
            if start == -1:
                break
            
            # Find corresponding closing }
            bracket_count = 1
            end = start + 1
            
            while end < length and bracket_count > 0:
                if text[end] == '{':
                    bracket_count += 1
                elif text[end] == '}':
                    bracket_count -= 1
                end += 1
            
            if bracket_count == 0:
                # Found complete JSON object
                try:
                    json_obj = text[start:end]
                    parsed = json.loads(json_obj)
                    result.append(parsed)
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON
            
            pos = end
        
        return result
    
    def read_new_logs(self):
        """Read new log data from file"""
        new_logs = []
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                # Move to last read position
                f.seek(self.last_position)
                new_content = f.read()
                
                # Update position
                self.last_position = f.tell()
                
                if new_content.strip():
                    # Extract JSON objects
                    json_objects = self.extract_json_objects(new_content)
                    
                    for json_obj in json_objects:
                        # Handle ElasticSearch response format
                        if 'hits' in json_obj and 'hits' in json_obj['hits']:
                            for hit in json_obj['hits']['hits']:
                                new_logs.append(hit)
                        elif '_source' in json_obj:
                            # Direct hit object
                            new_logs.append(json_obj)
                        else:
                            # Try using JSON object directly as source
                            new_logs.append({'_source': json_obj})
                            
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
            
        return new_logs
    
    def check_layer1_frequency_detection(self, source):
        try:
            timestamp_str = source.get("@timestamp")
            rule_data = source.get("rule", {})
            rule_id = rule_data.get("id")
            mitre_data = rule_data.get("mitre", {})
            mitre_id = mitre_data.get("id", [""])[0] if isinstance(mitre_data.get("id"), list) else mitre_data.get("id", "")

            if not timestamp_str or not rule_id:
                return False, None

            if mitre_id != 'T1548.003':
                return False, None

            try:
                from dateutil import parser
                timestamp = parser.parse(timestamp_str)
            except:
                return False, None

            current_record = {
                'timestamp': timestamp,
                'rule_id': rule_id,
                'mitre_id': mitre_id,
                'source': source
            }

            self.frequency_buffer.append(current_record)

            if len(self.frequency_buffer) < 2:
                return False, None

            return self.analyze_frequency_pattern_v2(list(self.frequency_buffer), current_record)

        except Exception as e:
            self.logger.error(f"Error in Layer 1 T1548.003 detection: {e}")
            return False, None

    def set_t1548_frequency_threshold(self, time_window_minutes=5, min_consecutive_occurrences=2):
        self.frequency_threshold = {
            'time_window_minutes': time_window_minutes,
            'min_consecutive_occurrences': min_consecutive_occurrences,
            'min_records': min_consecutive_occurrences 
        }
        self.logger.info(f"T1548.003 detection thresholds updated: {self.frequency_threshold}")
        self.logger.info(f"Will trigger alert when T1548.003 appears {min_consecutive_occurrences}+ times within {time_window_minutes} minutes")

    def analyze_frequency_pattern_v2(self, records, current_record):
        current_time = current_record['timestamp']
        current_mitre_id = current_record['mitre_id']
    
        if current_mitre_id != 'T1548.003':
            return False, None
    
        t1548_events = []
    
        for record in records:
            if record['mitre_id'] == 'T1548.003':
                time_diff = abs((current_time - record['timestamp']).total_seconds() / 60)
                if time_diff <= self.frequency_threshold['time_window_minutes']:
                    t1548_events.append(record)
    
        if len(t1548_events) >= 2:
            t1548_events.sort(key=lambda x: x['timestamp'])
        
            earliest_event = t1548_events[0]
            latest_event = t1548_events[-1]
        
            time_span = (latest_event['timestamp'] - earliest_event['timestamp']).total_seconds() / 60
        
            frequency_data = {
                'time_span_minutes': time_span,
                'time_diff_minutes': time_span,
                'event_count': len(t1548_events),
                'events_per_minute': len(t1548_events) / max(time_span, 1),
                'earliest_record': {
                    'source': earliest_event['source'],
                    'timestamp': earliest_event['timestamp'],
                    'firedtimes': earliest_event['source'].get('rule', {}).get('firedtimes', 0)
                },
                'latest_record': {
                    'source': latest_event['source'],
                    'timestamp': latest_event['timestamp'],
                    'firedtimes': latest_event['source'].get('rule', {}).get('firedtimes', 0)
                },
                'firedtimes_diff': latest_event['source'].get('rule', {}).get('firedtimes', 0) - earliest_event['source'].get('rule', {}).get('firedtimes', 0),
                'total_records_analyzed': len(t1548_events),
                'all_same_events': t1548_events,
                'detection_criteria': f"T1548.003 appeared {len(t1548_events)} times consecutively in {time_span:.2f} minutes"
            }
        
            self.logger.info(f"Layer 1 T1548.003 Detection Triggered:")
            self.logger.info(f"  MITRE ID: T1548.003")
            self.logger.info(f"  Consecutive occurrences: {len(t1548_events)} times in {time_span:.2f} minutes")
            self.logger.info(f"  Detection threshold: ≥2 consecutive occurrences")
        
            return True, frequency_data
    
        return False, None

    def generate_layer1_alert(self, source, frequency_data=None):
        """Generate Layer 1 alert for frequency detection with cooldown mechanism"""
        alert_type = "FREQUENCY_DETECTION"
        
        # Check cooldown first
        if self.is_layer1_alert_in_cooldown(alert_type):
            remaining_cooldown = self.layer1_alert_cooldown - (
                datetime.now() - self.layer1_active_alerts.get(alert_type, datetime.now())
            ).total_seconds()
            self.logger.debug(f"Layer 1 frequency alert in cooldown period (remaining: {remaining_cooldown:.0f}s)")
            return None
        
        # Update alert status
        self.update_layer1_alert_status(alert_type)
        
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]

        if frequency_data:
            latest_source = frequency_data['latest_record']['source']
            earliest_source = frequency_data['earliest_record']['source']
            time_diff = frequency_data['time_diff_minutes']
            consecutive_count = frequency_data['firedtimes_diff']
            firedtimes_diff = frequency_data.get('firedtimes_diff', 0)
        else:
            latest_source = source
            earliest_source = source
            time_diff = 0
            consecutive_count = 0
            firedtimes_diff = 0
    
        # Extract necessary fields
        agent_data = latest_source.get("agent", {})
        agent_ip = agent_data.get("ip", "unknown")
        agent_name = agent_data.get("name", "unknown")
        agent_id = agent_data.get("id", "unknown")

        rule_data = latest_source.get("rule", {})
        rule_id = rule_data.get("id", "unknown")
        rule_firedtimes = rule_data.get("firedtimes", 0)

        full_log = latest_source.get("full_log", latest_source.get("message", "unknown"))
        timestamp = latest_source.get("@timestamp", datetime.now().isoformat())
    
        # Print alert
        alert_message = f"""
============================================================
LAYER 1 CRITICAL ALERT - HIGH FREQUENCY DETECTION
============================================================
Time: {now_str}
Detection Layer: Layer 1 (Frequency-based Detection)
MITRE Technique: Method: High Frequency Activity Analysis
Confidence: 100% (Frequency Pattern Match)
Severity: CRITICAL

Frequency Analysis:
- Time Window: {time_diff:.2f} minutes
- Firedtimes Difference: {firedtimes_diff}
- Threshold: Time ≤ {self.frequency_threshold['time_window_minutes']} min AND consecutive occurrences ≥ {self.frequency_threshold.get('min_consecutive_occurrences', 2)}
- Records Analyzed: {frequency_data.get('total_records_analyzed', 0)}

Latest Event Information:
- Agent IP: {agent_ip}
- Agent Name: {agent_name}
- Agent ID: {agent_id}
- Rule ID: {rule_id}
- Rule Firedtimes: {rule_firedtimes}
- Timestamp: {timestamp}

Earliest Event Information:
- Timestamp: {frequency_data['earliest_record']['source'].get('@timestamp', 'unknown')}
- Firedtimes: {frequency_data['earliest_record']['firedtimes']}

Log Details:
- Latest Log: {full_log}
- Earliest Log: {frequency_data['earliest_record']['source'].get('full_log', 'unknown')}

============================================================
"""
    
        # Log alert
        self.logger.warning("LAYER 1 CRITICAL ALERT: HIGH FREQUENCY ACTIVITY DETECTED")
        self.logger.warning(alert_message)
    
        # Write to security alerts file
        with open('security_alerts.log', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} - LAYER 1 FREQUENCY ALERT\n")
            f.write(alert_message)
            f.write("\n" + "="*80 + "\n\n")
    
        # Update statistics
        self.stats['layer1_detections'] += 1
        self.stats['total_alerts'] += 1
    
        # Build alert result
        alert_result = {
            "detection_layer": 1,
            "is_alert": True,
            "scenario": "HIGH_FREQUENCY_ACTIVITY",
            "confidence": 1.0,
            "alert_type": "CRITICAL_FREQUENCY",
            "technique_id": "FREQUENCY_DETECTION",
            "frequency_analysis": {
                "consecutive_count": consecutive_count,
                "time_window_minutes": time_diff if frequency_data else 0,
                "detection_threshold": self.frequency_threshold.get('min_consecutive_occurrences', 2),
                "threshold_met": True
            },
            "raw_log": {
                "agent.ip": agent_ip,
                "agent.name": agent_name,
                "agent.id": agent_id,
                "rule.id": rule_id,
                "rule.firedtimes": rule_firedtimes,
                "rule.mitre.id": "T1548.003",
                "full_log": str(full_log),
                "@timestamp": timestamp,
            }
        }
    
        try:
            discord_success = self.discord_notifier.send_discord_alert(alert_result)
            if discord_success:
                self.logger.info("Layer 1 Discord alert sent successfully")
            else:
                self.logger.warning("Layer 1 Discord alert failed to send")
        except Exception as e:
            self.logger.error(f"Error sending Layer 1 Discord alert: {e}")
    
        return alert_result

    def preprocess_log_for_layer2(self, log_data):
        """Preprocess single log entry for Layer 2 analysis"""
        if not self.encoders:
            return None
            
        try:
            source = log_data.get('_source', {})
            
            # Extract necessary fields
            agent_info = source.get('agent', {})
            rule_info = source.get('rule', {})
            
            agent_ip = agent_info.get('ip', '192.168.1.1')
            agent_name = agent_info.get('name', 'unknown')
            agent_id = agent_info.get('id', '001')
            rule_id = rule_info.get('id', 1001)
            
            # Handle MITRE ID
            mitre_info = rule_info.get('mitre', {})
            if isinstance(mitre_info, dict):
                mitre_ids = mitre_info.get('id', ['T0000'])
            else:
                mitre_ids = ['T0000']
                
            full_log = source.get('full_log', source.get('message', 'unknown'))
            timestamp = source.get('@timestamp', datetime.now().isoformat())
            
            # Handle numeric features
            agent_ip_num = self._safe_extract_ip_number(agent_ip)
            agent_id_num = self._safe_extract_agent_id(agent_id)
            rule_id_num = self._safe_extract_rule_id(rule_id)
            
            # Handle MITRE ID (take first one)
            if isinstance(mitre_ids, list) and mitre_ids:
                mitre_id = mitre_ids[0]
            elif isinstance(mitre_ids, str):
                mitre_id = mitre_ids
            else:
                mitre_id = 'T0000'
            
            # Encode categorical features
            agent_name_encoded = self._safe_encode('agent_name', agent_name)
            eventdata_image_encoded = self._safe_encode('eventdata_image', full_log)
            mitre_id_encoded = self._safe_encode('mitre_id', mitre_id)
            
            # Build feature vector
            features = [
                agent_ip_num,
                agent_name_encoded,
                agent_id_num,
                eventdata_image_encoded,
                rule_id_num,
                mitre_id_encoded
            ]
            
            return {
                'features': features,
                'timestamp': timestamp,
                'original_data': source,
                'agent_name': agent_name,
                'agent_ip': agent_ip,
                'agent_id': agent_id,
                'rule_id': rule_id,
                'mitre_id': mitre_id,
                'full_log': str(full_log)
            }
            
        except Exception as e:
            self.logger.error(f"Error preprocessing log for Layer 2: {e}")
            return None
    
    def _safe_extract_ip_number(self, agent_ip):
        """Safely extract IP address numeric part"""
        try:
            if isinstance(agent_ip, str) and '.' in agent_ip:
                return int(agent_ip.split('.')[-1])
            else:
                return 1
        except:
            return 1
    
    def _safe_extract_agent_id(self, agent_id):
        """Safely extract Agent ID number"""
        try:
            if isinstance(agent_id, str) and agent_id.isdigit():
                return int(agent_id)
            elif isinstance(agent_id, str):
                numbers = re.findall(r'\d+', agent_id)
                return int(numbers[0]) if numbers else 1
            else:
                return int(agent_id) if agent_id else 1
        except:
            return 1
    
    def _safe_extract_rule_id(self, rule_id):
        """Safely extract Rule ID number"""
        try:
            return int(rule_id) if isinstance(rule_id, (int, str)) and str(rule_id).isdigit() else 1001
        except:
            return 1001
    
    def _safe_encode(self, feature_name, value):
        """Safely encode feature values, handle unseen values"""
        encoder = self.encoders.get(feature_name)
        if encoder is None:
            return 0
        
        try:
            str_value = str(value) if value is not None else 'unknown'
            return encoder.transform([str_value])[0]
        except ValueError:
            return 0
        except Exception as e:
            self.logger.warning(f"Encoding error for {feature_name}: {e}")
            return 0
    
    def check_layer2_special_rules(self, processed_log):
        """Layer 2: Check special attack rules (with cooldown mechanism)"""
        if not processed_log:
            return None, 0.0
        
        mitre_id = processed_log.get('mitre_id', '')
        
        # Add current MITRE ID to sequence
        self.mitre_sequence.append(mitre_id)
        
        # Check specific combination: T1548.003 + T1110
        if len(self.mitre_sequence) >= 2:
            recent_mitres = list(self.mitre_sequence)
            
            has_t1548_003 = 'T1548.003' in recent_mitres
            has_t1110 = 'T1110' in recent_mitres
            
            if has_t1548_003 and has_t1110:
                rule_name = "T1548.003_T1110_combination"
                if not self.is_special_rule_in_cooldown(rule_name):
                    self.update_special_rule_status(rule_name)
                    self.stats['layer2_special_rules'] += 1
                    return 'OT_RemoteAccess', 0.876
                else:
                    self.logger.debug(f"Special rule {rule_name} is in cooldown period")
                    return None, 0.0
        
        return None, 0.0
    
    def predict_attack_chain_layer2(self, sequence):
        """Layer 2: Use LSTM model to predict attack chain"""
        if not self.model:
            return None, 0.0
            
        try:
            if len(sequence) < self.sequence_length:
                return None, 0.0
            
            sequence_array = np.array(sequence, dtype=np.float32)
            X = sequence_array.reshape(1, self.sequence_length, -1)
            
            predictions = self.model.predict(X, verbose=0)
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.attack_chain_classes[predicted_class_idx]
            
            return predicted_class, confidence
            
        except Exception as e:
            self.logger.error(f"Error in Layer 2 prediction: {e}")
            return None, 0.0
    
    def is_alert_in_cooldown(self, attack_type):
        """Check if specific attack type is in cooldown period"""
        if attack_type not in self.active_alerts:
            return False
        
        last_alert_time = self.active_alerts[attack_type]
        current_time = datetime.now()
        time_diff = (current_time - last_alert_time).total_seconds()
        
        return time_diff < self.alert_cooldown
    
    def update_alert_status(self, attack_type):
        """Update alert status"""
        self.active_alerts[attack_type] = datetime.now()
        self.detected_attack_chains.add(attack_type)
    
    def is_special_rule_in_cooldown(self, rule_name):
        """Check if special rule is in cooldown period"""
        if rule_name not in self.last_special_rule_trigger:
            return False
        
        last_trigger_time = self.last_special_rule_trigger[rule_name]
        current_time = datetime.now()
        time_diff = (current_time - last_trigger_time).total_seconds()
        
        return time_diff < self.alert_cooldown
    
    def update_special_rule_status(self, rule_name):
        """Update special rule status"""
        self.last_special_rule_trigger[rule_name] = datetime.now()

    def is_layer1_alert_in_cooldown(self, alert_type="FREQUENCY_DETECTION"):
        """Check if Layer 1 frequency alert is in cooldown period"""
        if alert_type not in self.layer1_active_alerts:
            return False

        last_alert_time = self.layer1_active_alerts[alert_type]
        current_time = datetime.now()
        time_diff = (current_time - last_alert_time).total_seconds()

        return time_diff < self.layer1_alert_cooldown

    def update_layer1_alert_status(self, alert_type="FREQUENCY_DETECTION"):
        """Update Layer 1 alert status"""
        self.layer1_active_alerts[alert_type] = datetime.now()

    def generate_layer2_alert(self, predicted_class, confidence, recent_logs, trigger_log=None):
        """Generate Layer 2 alert message with detailed log information like Layer 1"""
        alert_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        severity_map = {
            'normal': 0,
            'IT_LSASS_Dump': 9,
            'IT_Powershell_based64': 8,
            'IT_Ransomware': 10,
            'OT_RemoteAccess': 7,
            'OT_CommandInjection': 8
        }
        
        severity = severity_map.get(predicted_class, 5)
        severity_label = "CRITICAL" if severity >= 9 else "HIGH" if severity >= 7 else "MEDIUM"
        
        # Get the trigger log (most recent log that caused the alert)
        trigger_log_info = trigger_log if trigger_log else (recent_logs[-1] if recent_logs else {})
        
        alert_message = f"""
============================================================
LAYER 2 SECURITY ALERT - {severity_label}
============================================================
Time: {alert_time}
Detection Layer: Layer 2 (LSTM Attack Chain Analysis)
Detected Attack Chain: {predicted_class}
Confidence Score: {confidence:.3f}
Alert Threshold: {self.threshold}
Severity Level: {severity}/10

Trigger Event Details:
- Agent IP: {trigger_log_info.get('agent_ip', 'unknown')}
- Agent Name: {trigger_log_info.get('agent_name', 'unknown')}
- Agent ID: {trigger_log_info.get('agent_id', 'unknown')}
- Rule ID: {trigger_log_info.get('rule_id', 'unknown')}
- MITRE ID: {trigger_log_info.get('mitre_id', 'unknown')}
- Timestamp: {trigger_log_info.get('timestamp', 'unknown')}
- Full Log: {trigger_log_info.get('full_log', 'unknown')}

Raw Log Data (JSON format):
{json.dumps({
    "agent.ip": trigger_log_info.get('agent_ip', 'unknown'),
    "agent.name": trigger_log_info.get('agent_name', 'unknown'),
    "agent.id": trigger_log_info.get('agent_id', 'unknown'),
    "rule.id": trigger_log_info.get('rule_id', 'unknown'),
    "rule.mitre.id": trigger_log_info.get('mitre_id', 'unknown'),
    "full_log": trigger_log_info.get('full_log', 'unknown'),
    "@timestamp": trigger_log_info.get('timestamp', 'unknown')
}, indent=2)}

Recent Activity Pattern ({len(recent_logs)} entries):
"""
        
        for i, log_entry in enumerate(recent_logs[-5:], 1):
            alert_message += f"""
Log Entry {i}:
- Time: {log_entry.get('timestamp', 'Unknown')}
- Agent: {log_entry.get('agent_name', 'Unknown')} ({log_entry.get('agent_ip', 'Unknown')})
- Agent ID: {log_entry.get('agent_id', 'Unknown')}
- Rule ID: {log_entry.get('rule_id', 'Unknown')}
- MITRE ID: {log_entry.get('mitre_id', 'Unknown')}
- Log: {log_entry.get('full_log', 'Unknown')[:100]}...
"""
        
        alert_message += f"\n============================================================\n"
        
        return alert_message
    
    def generate_layer2_alert_result(self, predicted_class, confidence, trigger_log):
        """Generate Layer 2 alert result structure similar to Layer 1"""
        return {
            "detection_layer": 2,
            "is_alert": True,
            "scenario": predicted_class,
            "confidence": confidence,
            "alert_type": "LSTM_ATTACK_CHAIN",
            "technique_id": trigger_log.get('mitre_id', 'unknown'),
            "raw_log": {
                "agent.ip": trigger_log.get('agent_ip', 'unknown'),
                "agent.name": trigger_log.get('agent_name', 'unknown'),
                "agent.id": trigger_log.get('agent_id', 'unknown'),
                "rule.id": trigger_log.get('rule_id', 'unknown'),
                "rule.mitre.id": trigger_log.get('mitre_id', 'unknown'),
                "full_log": trigger_log.get('full_log', 'unknown'),
                "@timestamp": trigger_log.get('timestamp', 'unknown')
            }
        }
    
    def process_new_logs(self):
        """Main processing function - implements two-layer detection"""
        try:
            new_logs = self.read_new_logs()
            
            if new_logs:
                self.logger.info(f"Processing {len(new_logs)} new log entries")
                
                processed_count = 0
                layer1_hits = 0
                layer2_processed = 0
                
                for log_entry in new_logs:
                    self.stats['total_processed'] += 1
                    processed_count += 1
                    
                    source = log_entry.get('_source', {})
                    
                    is_frequency_alert, frequency_data = self.check_layer1_frequency_detection(source)
                    
                    if is_frequency_alert:
                        layer1_hits += 1
                        self.logger.info(f"LAYER 1 HIT: High frequency activity detected")
                
                        alert_result = self.generate_layer1_alert(source, frequency_data)
                
                        if alert_result:  # Only log if alert was actually generated (not in cooldown)
                            self.logger.warning(f"LAYER 1 ALERT: {alert_result}")
                        else:
                            self.logger.debug("Layer 1 alert suppressed due to cooldown period")
                
                        continue

                    layer2_processed += 1
                    processed_log = self.preprocess_log_for_layer2(log_entry)
                    
                    if processed_log:
                        # Add to sequence buffer
                        self.sequence_buffer.append(processed_log['features'])
                        self.processed_logs.append(processed_log)
                        
                        # Check Layer 2 special rules
                        special_class, special_confidence = self.check_layer2_special_rules(processed_log)
                        
                        predicted_class = None
                        confidence = 0.0
                        should_alert = False
                        trigger_log = processed_log  # Current log is the trigger
                        
                        if special_class:
                            # Use special rule result
                            predicted_class = special_class
                            confidence = special_confidence
                            should_alert = True
                            self.logger.info(f"LAYER 2 SPECIAL RULE: {predicted_class} (confidence: {confidence:.3f})")
                        else:
                            # Normal LSTM model prediction when buffer is full
                            if len(self.sequence_buffer) == self.sequence_length:
                                sequence = list(self.sequence_buffer)
                                predicted_class, confidence = self.predict_attack_chain_layer2(sequence)
                                
                                if predicted_class:
                                    self.logger.info(f"LAYER 2 LSTM: {predicted_class} (confidence: {confidence:.3f})")
                                    if confidence >= self.threshold and predicted_class != 'normal':
                                        if not self.is_alert_in_cooldown(predicted_class):
                                            should_alert = True
                        
                        # Handle Layer 2 alerts
                        if should_alert and predicted_class and confidence >= self.threshold and predicted_class != 'normal':
                            # Update alert status
                            self.update_alert_status(predicted_class)
                            
                            # Generate alert with detailed information
                            alert_message = self.generate_layer2_alert(
                                predicted_class, 
                                confidence, 
                                list(self.processed_logs),
                                trigger_log
                            )
                            
                            # Generate alert result structure
                            alert_result = self.generate_layer2_alert_result(
                                predicted_class,
                                confidence,
                                trigger_log
                            )
                            
                            # Log alert
                            self.logger.warning("LAYER 2 ATTACK DETECTED!")
                            self.logger.warning(alert_message)
                            self.logger.warning(f"LAYER 2 ALERT RESULT: {alert_result}")
                            
                            # Write to security alerts file
                            with open('security_alerts.log', 'a', encoding='utf-8') as f:
                                f.write(f"{datetime.now().isoformat()} - LAYER 2 ALERT\n")
                                f.write(alert_message)
                                f.write(f"\nAlert Result: {json.dumps(alert_result, indent=2)}\n")
                                f.write("\n" + "="*80 + "\n\n")
                            
                            try:
                                discord_success = self.discord_notifier.send_discord_alert(alert_result)
                                if discord_success:
                                    self.logger.info("Layer 2 Discord alert sent successfully")
                                else:
                                    self.logger.warning("Layer 2 Discord alert failed to send")
                            except Exception as e:
                                self.logger.error(f"Error sending Layer 2 Discord alert: {e}")
                            
                            self.stats['layer2_detections'] += 1
                            self.stats['total_alerts'] += 1
                            
                        elif predicted_class and confidence >= self.threshold and predicted_class != 'normal':
                            # In cooldown period
                            remaining_cooldown = self.alert_cooldown - (datetime.now() - self.active_alerts.get(predicted_class, datetime.now())).total_seconds()
                            self.logger.debug(f"Layer 2 attack {predicted_class} detected but in cooldown period (remaining: {remaining_cooldown:.0f}s)")
                        elif predicted_class:
                            self.logger.debug(f"Layer 2 normal activity: {predicted_class} (confidence: {confidence:.3f})")
                
                # Log processing summary
                if processed_count > 0:
                    self.logger.info(f"Processing Summary:")
                    self.logger.info(f"  - Total processed: {processed_count}")
                    self.logger.info(f"  - Layer 1 (Frequency Detection) hits: {layer1_hits}")
                    self.logger.info(f"  - Layer 2 processed: {layer2_processed}")
                    self.logger.info(f"  - Detection Statistics: {self.stats}")
                            
        except Exception as e:
            self.logger.error(f"Error processing new logs: {e}")
            import traceback
            traceback.print_exc()
    
    def set_frequency_threshold(self, time_window_minutes=5, firedtimes_diff_threshold=2, min_records=2):
        self.frequency_threshold = {
            'time_window_minutes': time_window_minutes,
            'min_occurrences': min_occurrences,
            'min_records': min_records
        }
        self.logger.info(f"Frequency detection thresholds updated: {self.frequency_threshold}")

    def get_frequency_threshold(self):
            return self.frequency_threshold.copy()
    
    def get_detection_statistics(self):
        """Get current detection statistics"""
        current_time = datetime.now()
        layer2_active_cooldowns = 0
        layer1_active_cooldowns = 0
        
        for attack_type, last_alert_time in self.active_alerts.items():
            time_diff = (current_time - last_alert_time).total_seconds()
            if time_diff < self.alert_cooldown:
                layer2_active_cooldowns += 1

        for alert_type, last_alert_time in self.layer1_active_alerts.items():
            time_diff = (current_time - last_alert_time).total_seconds()
            if time_diff < self.layer1_alert_cooldown:
                layer1_active_cooldowns += 1
        
        return {
            **self.stats,
            'layer1_active_cooldowns': layer1_active_cooldowns,
            'layer2_active_cooldowns': layer2_active_cooldowns,
            'total_active_cooldowns': layer1_active_cooldowns + layer2_active_cooldowns,
            'detected_attack_types': list(self.detected_attack_chains),
            'total_unique_attacks': len(self.detected_attack_chains),
            'layer1_cooldown_period': self.layer1_alert_cooldown,
            'layer2_cooldown_period': self.alert_cooldown
        }

    def set_layer1_cooldown(self, cooldown_seconds):
        """Set Layer 1 alert cooldown period"""
        self.layer1_alert_cooldown = cooldown_seconds
        self.logger.info(f"Layer 1 alert cooldown period updated to: {cooldown_seconds} seconds")
    
    def get_layer1_cooldown_status(self):
        """Get Layer 1 cooldown status"""
        current_time = datetime.now()
        cooldown_info = {}
        
        for alert_type, last_alert_time in self.layer1_active_alerts.items():
            time_diff = (current_time - last_alert_time).total_seconds()
            remaining_time = max(0, self.layer1_alert_cooldown - time_diff)
            cooldown_info[alert_type] = {
                'last_alert_time': last_alert_time.isoformat(),
                'remaining_cooldown_seconds': remaining_time,
                'is_in_cooldown': remaining_time > 0
            }
        
        return cooldown_info

    def run_file_monitoring(self):
        """Run file monitoring mode"""
        self.logger.info(f"Starting Integrated Two-Layer Security Detection")
        self.logger.info(f"Monitoring file: {self.log_file_path}")
        self.logger.info(f"Layer 1: Frequency Critical Detection (Direct Match)")
        self.logger.info(f"Layer 2: LSTM Attack Chain Analysis (Threshold: {self.threshold})")
        
        # Process existing logs first
        self.logger.info("Processing existing logs...")
        self.process_new_logs()
        
        # Setup file monitoring
        try:
            event_handler = LogFileHandler(self)
            observer = Observer()
            
            watch_dir = os.path.dirname(self.log_file_path)
            if not watch_dir:
                watch_dir = '.'
                
            observer.schedule(event_handler, watch_dir, recursive=False)
            observer.start()
            
            self.logger.info(f"File monitoring started for directory: {watch_dir}")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Stopping monitoring by user request")
                stats = self.get_detection_statistics()
                self.logger.info(f"Final detection statistics: {stats}")
                observer.stop()
            
            observer.join()
            
        except ImportError:
            self.logger.warning("watchdog library not found. Using polling mode instead.")
            self.run_polling_mode(10)
    
    def run_polling_mode(self, polling_interval=10):
        """Run polling mode (periodically check file changes)"""
        self.logger.info(f"Starting polling mode with {polling_interval} second intervals")
        self.logger.info(f"Layer 1: Frequency Critical Detection")
        self.logger.info(f"Layer 2: LSTM Attack Chain Analysis")
        
        try:
            while True:
                self.process_new_logs()
                time.sleep(polling_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping polling mode by user request")
            stats = self.get_detection_statistics()
            self.logger.info(f"Final detection statistics: {stats}")

def main():
    """Main function"""
    # Configuration parameters
    config = {
        'log_file_path': '/home/danish/Realtime_LSTM_Chain_Detection/data/wazuh_log.log',
        'model_path': 'model/lstm_attack_chain_model.keras',
        'encoders_path': 'encoders.pkl', 
        'classes_path': 'classes',
        'threshold': 0.7,  # Alert threshold for Layer 2
        'sequence_length': 10,
        'detection_log': 'integrated_security_detection.log',
        'alert_cooldown': 100,  # Cooldown period in seconds
        'discord_webhook_url': 'https://discordapp.com/api/webhooks/1375154135967727637/Jetux5lMuuF8TI1CNgp66VVxjvRyS_RgkfCgQu-IwJe0xCZOWO36WR-stZm3iPjSKboK',
            'enable_discord_alerts': True
    }
    
    # Create integrated detector instance
    detector = IntegratedSecurityDetector(**config)
    
    # Choose running mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--polling':
        # Polling mode
        polling_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        detector.run_polling_mode(polling_interval)
    else:
        try:
            detector.run_file_monitoring()
        except ImportError:
            print("watchdog library not found. Using polling mode instead.")
            print("Install with: pip install watchdog")
            detector.run_polling_mode(10)

if __name__ == "__main__":
    main()
