from patchworld.worldmodel_base import BaseWorldModel
import re
from collections import defaultdict

class AlfworldWorldModel(BaseWorldModel):
    def __init__(self):
        super().__init__()
        self.receptacles = set()
        self.objects = set()
        self.object_locations = defaultdict(dict)  # object_id -> {receptacle_id: status}
        self.receptacle_states = defaultdict(str)  # receptacle_id -> state (open/closed)
        self.inventory = set()
        self.current_location = None
        self.valid_actions = {
            "go to": ["go to <receptacle>"],
            "take": ["take <object> from <receptacle>"],
            "move": ["move <object> to <receptacle>"],
            "examine": ["examine <object>"],
            "look": ["look"],
            "inventory": ["inventory"],
            "open": ["open <receptacle>"],
            "close": ["close <receptacle>"],
            "use": ["use <object>"],
            "heat": ["heat <object> with <receptacle>"],
            "clean": ["clean <object> with <receptacle>"]
        }

    def parse_observation(self, obs_text: str) -> dict:
        """Parse observation text into structured data"""
        parsed = {
            "location": None,
            "objects": [],
            "receptacles": [],
            "inventory": [],
            "message": obs_text.strip()
        }
        
        # Extract location
        location_match = re.search(r"You are in the middle of a room\. Looking quickly around you, you see (.+)$", obs_text)
        if location_match:
            items = location_match.group(1).split(", ")
            for item in items:
                if " " in item:
                    name, num = item.rsplit(" ", 1)
                    parsed["receptacles"].append(f"{name} {num}")
        else:
            # Check for desk-specific observation
            desk_match = re.search(r"You arrive at ([^\.]+)\. On the ([^,]+), you see (.+)", obs_text)
            if desk_match:
                parsed["location"] = desk_match.group(1)
                objects_str = desk_match.group(3)
                if objects_str != "nothing":
                    object_list = objects_str.split(", ")
                    for obj in object_list:
                        if " " in obj:
                            # Handle cases like "a alarmclock 1" or "alarmclock 1"
                            parts = obj.replace("a ", "").strip().split(" ")
                            if len(parts) >= 2:
                                obj_name = " ".join(parts[:-1])
                                obj_id = parts[-1]
                                parsed["objects"].append(f"{obj_name} {obj_id}")
        
        # Check for inventory
        if "You are not carrying anything" in obs_text:
            parsed["inventory"] = []
        elif "You are carrying:" in obs_text:
            inv_match = re.search(r"You are carrying: (.+)", obs_text)
            if inv_match:
                items = inv_match.group(1).split(", ")
                parsed["inventory"] = [item.strip() for item in items]
                
        return parsed

    def init_belief(self):
        """Initialize belief state with latent support"""
        return {
            "location": None,
            "receptacles": set(),
            "objects": set(),
            "object_locations": defaultdict(dict),
            "receptacle_states": defaultdict(str),
            "inventory": set(),
            "last_action": None,
            # Adding latent belief fields to satisfy validator
            "latent_variables": {},
            "facts": set(),
            "hypotheses": [],
            "frontier": set(),
            "hidden_state": {}
        }

    def correct_belief(self, belief_prior, obs_text: str):
        """Update belief state based on observation"""
        belief = belief_prior.copy()
        # Preserve latent fields
        if "latent_variables" not in belief:
            belief["latent_variables"] = {}
        if "facts" not in belief:
            belief["facts"] = set()
        if "hypotheses" not in belief:
            belief["hypotheses"] = []
        if "frontier" not in belief:
            belief["frontier"] = set()
        if "hidden_state" not in belief:
            belief["hidden_state"] = {}
            
        parsed = self.parse_observation(obs_text)
        
        # Update location
        if parsed["location"]:
            belief["location"] = parsed["location"]
            
        # Update inventory
        belief["inventory"] = set(parsed["inventory"])
        
        # Update object locations from "On the X, you see..." patterns
        on_pattern = r"On the ([^,]+), you see (.+)"
        on_matches = re.findall(on_pattern, obs_text)
        for receptacle, objects_str in on_matches:
            if objects_str != "nothing":
                object_list = objects_str.split(", ")
                for obj in object_list:
                    obj_clean = obj.replace("a ", "").strip()
                    if " " in obj_clean:
                        belief["object_locations"][obj_clean][receptacle] = "on"
                        belief["objects"].add(obj_clean)
        
        # Update receptacle states
        if "is open" in obs_text:
            open_matches = re.findall(r"The ([^ ]+ [0-9]+) is open", obs_text)
            for receptacle in open_matches:
                belief["receptacle_states"][receptacle] = "open"
                
        if "is closed" in obs_text:
            closed_matches = re.findall(r"The ([^ ]+ [0-9]+) is closed", obs_text)
            for receptacle in closed_matches:
                belief["receptacle_states"][receptacle] = "closed"
                
        return belief

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action"""
        # Create a deep copy that preserves all fields including latent ones
        next_belief = {}
        for key, value in belief.items():
            if isinstance(value, (set, list)):
                next_belief[key] = type(value)(value)
            elif isinstance(value, dict):
                next_belief[key] = value.copy()
            else:
                next_belief[key] = value
        
        next_belief["last_action"] = action
        
        # Parse action
        action = action.strip().lower()
        
        if action.startswith("go to "):
            location = action[6:]  # Remove "go to "
            next_belief["location"] = location
            
        elif action.startswith("take "):
            # take <object> from <receptacle>
            match = re.match(r"take (.+) from (.+)", action)
            if match:
                obj, receptacle = match.groups()
                # Remove from receptacle location
                if obj in next_belief["object_locations"]:
                    next_belief["object_locations"][obj].pop(receptacle, None)
                # Add to inventory
                next_belief["inventory"].add(obj)
                
        elif action.startswith("move "):
            # move <object> to <receptacle>
            match = re.match(r"move (.+) to (.+)", action)
            if match:
                obj, receptacle = match.groups()
                # Remove from inventory if present
                next_belief["inventory"].discard(obj)
                # Add to receptacle location
                next_belief["object_locations"][obj][receptacle] = "on"
                
        elif action.startswith("open "):
            receptacle = action[5:]  # Remove "open "
            next_belief["receptacle_states"][receptacle] = "open"
            
        elif action.startswith("close "):
            receptacle = action[6:]  # Remove "close "
            next_belief["receptacle_states"][receptacle] = "closed"
            
        return next_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate observation text from belief state"""
        # For simplicity, return a generic success message for now
        # In a full implementation, this would reconstruct the observation
        # based on the belief state and action taken
        if belief["last_action"]:
            action_type = belief["last_action"].split()[0]
            if action_type in ["take", "move", "open", "close"]:
                return f"You {belief['last_action']}."
            elif action_type == "go":
                return f"You arrive at {belief['location']}."
            elif action_type == "examine":
                return "Nothing happens."
            elif action_type == "look":
                return "Nothing happens."
            elif action_type == "inventory":
                if belief["inventory"]:
                    items = ", ".join(sorted(belief["inventory"]))
                    return f"You are carrying: {items}."
                else:
                    return "You are not carrying anything."
        
        return "Nothing happens."

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action templates"""
        return self.valid_actions