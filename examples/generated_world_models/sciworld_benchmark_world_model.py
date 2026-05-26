from patchworld.worldmodel_base import BaseWorldModel
import re

class SciworldWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """
        Parse observation text into structured data.
        """
        # Basic parsing - extract key elements
        parsed = {
            "raw_text": obs_text,
            "room": "",
            "objects": [],
            "inventory": [],
            "actions": [],
            "task": "",
            "room_contents": {},
            "object_states": {},
            "container_contents": {}
        }
        
        # Extract room name
        room_match = re.search(r"This room is called the ([^.]+)\.", obs_text)
        if room_match:
            parsed["room"] = room_match.group(1).strip()
            
        # Extract task description
        task_match = re.search(r"Task description:\s*([^.]+(?:\.[^.]+)*)", obs_text)
        if task_match:
            parsed["task"] = task_match.group(1)
            
        # Extract inventory
        inv_match = re.search(r"In your inventory, you see:\s*((?:\s*.+)+?)(?=\n\n|\Z)", obs_text)
        if inv_match:
            items = inv_match.group(1).strip().split("\n")
            parsed["inventory"] = [item.strip() for item in items if item.strip() and item.strip() != "nothing"]
        elif "In your inventory, you see:" in obs_text and "nothing" in obs_text:
            parsed["inventory"] = []
            
        # Extract room contents
        contents_match = re.search(r"In it, you see:((?:\n\t.+(?:\n\t\t.+)*)+)", obs_text)
        if contents_match:
            contents_text = contents_match.group(1)
            # Parse objects and their states/contents
            lines = contents_text.strip().split('\n')
            current_parent = None
            for line in lines:
                if line.startswith('\t\t'):
                    # This is content of a container
                    if current_parent:
                        content = line.strip()
                        if current_parent not in parsed["container_contents"]:
                            parsed["container_contents"][current_parent] = []
                        parsed["container_contents"][current_parent].append(content)
                elif line.startswith('\t'):
                    # This is an object
                    obj_desc = line.strip()
                    obj_name = obj_desc.split('.')[0] if '.' in obj_desc else obj_desc
                    parsed["objects"].append(obj_name)
                    current_parent = obj_name
                    
                    # Check for states like open/closed, on/off
                    if 'open' in obj_desc:
                        parsed["object_states"][obj_name] = 'open'
                    elif 'closed' in obj_desc:
                        parsed["object_states"][obj_name] = 'closed'
                    elif 'on' in obj_desc:
                        parsed["object_states"][obj_name] = 'on'
                    elif 'off' in obj_desc:
                        parsed["object_states"][obj_name] = 'off'
                        
        return parsed

    def init_belief(self):
        """
        Initialize belief state with latent support.
        """
        return {
            "room": "unknown",
            "objects": {},
            "inventory": [],
            "task": "",
            "room_contents": {},
            "object_states": {},
            "container_contents": {},
            "latent_variables": {
                "door_states": {},
                "container_states": {},
                "object_locations": {}
            },
            "facts": set(),
            "hypotheses": [],
            "frontier": set(),
            "hidden_state": {}
        }

    def correct_belief(self, belief_prior, obs_text: str):
        """
        Update belief state based on new observation.
        """
        parsed = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        if parsed["room"]:
            # Fix room name normalization issue
            room_name = parsed["room"].strip()
            if room_name.startswith("the "):
                room_name = room_name[4:]
            elif room_name.startswith("LOC "):
                room_name = room_name[4:]
            belief["room"] = room_name
            
        if parsed["task"]:
            belief["task"] = parsed["task"]
            
        if parsed["inventory"]:
            belief["inventory"] = parsed["inventory"].copy()
        elif "In your inventory, you see:" in obs_text and "nothing" in obs_text:
            belief["inventory"] = []
        # Only clear inventory if explicitly stated as empty, otherwise preserve it
            
        # Update room contents
        if parsed["objects"]:
            belief["room_contents"] = {obj: True for obj in parsed["objects"]}
            
        # Update object states
        belief["object_states"].update(parsed["object_states"])
        
        # Update container contents
        belief["container_contents"].update(parsed["container_contents"])
        
        # Maintain latent variables
        if "latent_variables" not in belief:
            belief["latent_variables"] = {
                "door_states": {},
                "container_states": {},
                "object_locations": {}
            }
            
        if "facts" not in belief:
            belief["facts"] = set()
            
        if "hypotheses" not in belief:
            belief["hypotheses"] = []
            
        if "frontier" not in belief:
            belief["frontier"] = set()
            
        if "hidden_state" not in belief:
            belief["hidden_state"] = {}
            
        return belief

    def predict_belief(self, belief, action: str):
        """
        Predict next belief state given current belief and action.
        """
        belief_next = belief.copy()
        
        # Handle specific actions that change state
        if action.startswith("open "):
            obj_name = action[5:]  # Remove "open "
            if obj_name not in belief_next["object_states"]:
                belief_next["object_states"][obj_name] = "open"
            else:
                belief_next["object_states"][obj_name] = "open"
                
        elif action.startswith("pick up "):
            obj_name = action[8:]  # Remove "pick up "
            if obj_name not in belief_next["inventory"]:
                belief_next["inventory"].append(obj_name)
            # Remove from room contents if present
            if obj_name in belief_next.get("room_contents", {}):
                del belief_next["room_contents"][obj_name]
                
        elif action.startswith("put down "):
            obj_name = action[9:]  # Remove "put down "
            if obj_name in belief_next["inventory"]:
                belief_next["inventory"].remove(obj_name)
                
        elif action.startswith("go to "):
            room_name = action[6:]  # Remove "go to "
            # Fix room name normalization
            if room_name.startswith("the "):
                room_name = room_name[4:]
            elif room_name.startswith("LOC "):
                room_name = room_name[4:]
            belief_next["room"] = room_name
            
        elif action.startswith("examine ") or action.startswith("look at "):
            # These actions can change the current room context
            location = action[8:] if action.startswith("examine ") else action[8:]  # Remove "examine " or "look at "
            # If it's a room name, update the current room
            rooms = ["kitchen", "hallway", "greenhouse", "bathroom", "living room", "bedroom", "workshop", "art studio", "foundry"]
            # Normalize location name
            loc_clean = location
            if loc_clean.startswith("the "):
                loc_clean = loc_clean[4:]
            if loc_clean in rooms:
                belief_next["room"] = loc_clean
                
        # Ensure all required latent fields exist
        required_fields = ["latent_variables", "facts", "hypotheses", "frontier", "hidden_state"]
        for field in required_fields:
            if field not in belief_next:
                if field == "latent_variables":
                    belief_next[field] = {
                        "door_states": {},
                        "container_states": {},
                        "object_locations": {}
                    }
                elif field == "facts":
                    belief_next[field] = set()
                elif field == "hypotheses":
                    belief_next[field] = []
                elif field == "frontier":
                    belief_next[field] = set()
                elif field == "hidden_state":
                    belief_next[field] = {}
                    
        return belief_next

    def readout_observation(self, belief, action: str = "") -> str:
        """
        Generate observation text from belief state.
        """
        if action.startswith("look around") or action == "look around":
            # Generate detailed room description
            room_name = belief.get('room', 'unknown')
            obs_lines = [f"This room is called the {room_name}."]
            
            if belief.get("room_contents") or belief.get("inventory"):
                obs_lines.append("In it, you see:")
                
                # Add room objects
                for obj_name in belief.get("room_contents", {}):
                    obj_line = f"\t{obj_name}"
                    if obj_name in belief.get("object_states", {}):
                        state = belief["object_states"][obj_name]
                        if state in ["open", "closed"]:
                            obj_line += f". The {obj_name} is {state}."
                        elif state in ["on", "off"]:
                            obj_line += f", which is turned {state}."
                    
                    # Check for container contents
                    if obj_name in belief.get("container_contents", {}):
                        contents = belief["container_contents"][obj_name]
                        if contents:
                            obj_line += f". On the {obj_name} is: {', '.join(contents)}."
                        else:
                            obj_line += f". On the {obj_name} is: nothing."
                    elif any(keyword in obj_name for keyword in ["drawer", "cupboard", "fridge", "oven", "freezer"]):
                        obj_line += ". The door is closed."
                        
                    obs_lines.append(obj_line)
                
                # Always add the agent to room contents when looking around
                obs_lines.append("\tthe agent")
                    
            return "\n".join(obs_lines)
            
        elif action.startswith("inventory") or action == "inventory":
            if belief.get("inventory"):
                items = "\n\t".join(belief["inventory"])
                return f"In your inventory, you see:\n\t{items}"
            else:
                return "In your inventory, you see:\n\tnothing"
                
        elif action.startswith("task") or action == "task":
            if belief.get("task"):
                return f"Task description:\n{belief['task']}"
            else:
                return "No task specified."
                
        elif action.startswith("open "):
            obj_name = action[5:]  # Remove "open "
            return f"The {obj_name} is now open."
            
        elif action.startswith("look in "):
            container_name = action[8:]  # Remove "look in "
            if container_name in belief.get("container_contents", {}):
                contents = belief["container_contents"][container_name]
                if contents:
                    content_list = "\n\t".join(contents)
                    return f"Inside the {container_name} is: \n\t{content_list}"
                else:
                    return f"Inside the {container_name} is: \n\tnothing"
            else:
                return f"Inside the {container_name} is: \n\tnothing"
                
        elif action.startswith("pick up "):
            obj_name = action[8:]  # Remove "pick up "
            return f"You move the {obj_name} to the inventory."
            
        elif action.startswith("focus on "):
            obj_name = action[9:]  # Remove "focus on "
            return f"You focus on the {obj_name}."
            
        elif action.startswith("look at "):
            obj_name = action[8:]  # Remove "look at "
            # Try to find object description in belief
            if obj_name in belief.get("object_states", {}):
                state = belief["object_states"][obj_name]
                if "door" in obj_name:
                    return f"A door to the {obj_name.replace(' door', '')} (that is {state})"
                else:
                    return f"a {obj_name}, which is turned {state}."
            elif obj_name in ["kitchen", "hallway", "greenhouse", "bathroom", "living room", "bedroom", "workshop", "art studio", "foundry"]:
                # Looking at a room
                return f"This room is called the {obj_name}."
            else:
                return f"a {obj_name}"
                
        elif action.startswith("examine "):
            obj_name = action[8:]  # Remove "examine "
            if obj_name in ["kitchen", "hallway", "greenhouse", "bathroom", "living room", "bedroom", "workshop", "art studio", "foundry"]:
                # Examining a room - return basic room description
                return f"This room is called the {obj_name}."
            else:
                return f"a {obj_name}"
                
        elif action.startswith("go to "):
            room_name = action[6:]  # Remove "go to "
            # Normalize room name
            if room_name.startswith("the "):
                room_name = room_name[4:]
            return f"You move to the {room_name}."
            
        elif action == "wait" or action == "wait1":
            return "You decide to wait for 10 iterations."
            
        elif action == "":
            # Default response for ambiguous situations
            return "Ambiguous request: Please enter the number for the action you intended (or blank to cancel):"
            
        else:
            # For unhandled actions, return a generic but non-empty response
            return f"You perform the action: {action}"

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """
        Return valid action templates.
        """
        return {
            "look at": ["look at <object>"],
            "examine": ["examine <object>"],
            "focus on": ["focus on <object>"],
            "pick up": ["pick up <object>"],
            "put down": ["put down <object>"],
            "move": ["move <object> to <location>"],
            "go to": ["go to <location>"],
            "open": ["open <object>"],
            "close": ["close <object>"],
            "activate": ["activate <object>"],
            "deactivate": ["deactivate <object>"],
            "connect": ["connect <object> to <object>"],
            "disconnect": ["disconnect <object>"],
            "use": ["use <object> on <object>"],
            "pour": ["pour <object> in <object>"],
            "dunk": ["dunk <object> in <object>"],
            "mix": ["mix <object>"],
            "read": ["read <object>"],
            "inventory": ["inventory"],
            "task": ["task"],
            "look around": ["look around"],
            "wait": ["wait"],
            "wait1": ["wait1"],
            "look in": ["look in <object>"]
        }