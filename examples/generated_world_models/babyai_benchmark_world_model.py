from patchworld.worldmodel_base import BaseWorldModel
import re
from collections import defaultdict

class BabyaiWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse observation text into structured data"""
        result = {
            'objects': [],
            'carrying': None,
            'facing': None,
            'goal': None,
            'walls': True
        }
        
        # Extract goal if present
        goal_match = re.search(r'Your goal: (.+)', obs_text)
        if goal_match:
            result['goal'] = goal_match.group(1)
        
        # Extract carrying information
        carrying_match = re.search(r'You are carrying (?:a|an|the) ([^.]+)', obs_text)
        if carrying_match:
            result['carrying'] = carrying_match.group(1)
        elif 'You are not carrying anything' in obs_text:
            result['carrying'] = None
            
        # Extract facing information
        facing_match = re.search(r'You are facing (?:a|an|the) ([^.]+)', obs_text)
        if facing_match:
            result['facing'] = facing_match.group(1)
        elif (facing_wall := re.search(r'You are facing a wall(?: (\d+) steps away)?', obs_text)):
            steps = facing_wall.group(1) if facing_wall.group(1) else "unknown"
            result['facing'] = f"wall {steps} steps away"
        elif 'You are facing a wall' in obs_text:
            result['facing'] = "wall"
            
        # Extract objects - more robust pattern
        if "In front of you in this room, you can see several objects:" in obs_text:
            object_section = obs_text.split("In front of you in this room, you can see several objects:")[1]
            if "You are facing" in object_section:
                object_section = object_section.split("You are facing")[0]
            
            # Handle the case where objects are listed
            object_lines = object_section.strip()
            
            # Pattern to match objects like: "There is a red box 1 1 steps in front of you and 2 steps to your left."
            object_pattern = r'There is (?:a|an) ([^,]+?) (\d+)(?: right in front of you )?(?:(\d+) steps away|(\d+) steps in front of you)?(?: and (\d+) steps to your (left|right))?[,.]'
            
            for match in re.finditer(object_pattern, object_lines):
                obj_name = match.group(1).strip()
                obj_id = match.group(2)
                
                # Determine front steps
                front_steps = "0"
                if match.group(3):  # steps away
                    front_steps = match.group(3)
                elif match.group(4):  # steps in front of you
                    front_steps = match.group(4)
                elif 'right in front of you' in match.group(0):
                    front_steps = "0"
                
                # Determine lateral position
                lateral_steps = 0
                lateral_dir = 'center'
                if match.group(5) and match.group(6):  # Has lateral position
                    lateral_steps = int(match.group(5))
                    lateral_dir = match.group(6)
                
                full_name = f"{obj_name} {obj_id}"
                result['objects'].append({
                    'name': obj_name,
                    'id': obj_id,
                    'full_name': full_name,
                    'front_steps': int(front_steps),
                    'lateral_steps': lateral_steps,
                    'lateral_dir': lateral_dir
                })
        
        return result

    def init_belief(self):
        """Initialize belief state with latent support"""
        return {
            'objects': [],
            'carrying': None,
            'facing': None,
            'position': (0, 0),  # (front, lateral) coordinates
            'orientation': 0,    # 0=north, 1=east, 2=south, 3=west
            'goal': None,
            'latent_variables': {},  # Add latent support
            'facts': set(),         # Track known facts
            'hypotheses': {},       # Track hypotheses about hidden state
            'frontier': set(),      # Track frontier of exploration
            'hidden_state': {}      # General hidden state tracking
        }

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct belief state based on observation"""
        parsed = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        # Deep copy lists and dicts
        belief['objects'] = [obj.copy() for obj in parsed['objects']]
        belief['carrying'] = parsed['carrying']
        belief['facing'] = parsed['facing']
        belief['goal'] = parsed['goal']
        
        # Maintain latent variables
        if 'latent_variables' not in belief:
            belief['latent_variables'] = {}
        if 'facts' not in belief:
            belief['facts'] = set()
        if 'hypotheses' not in belief:
            belief['hypotheses'] = {}
        if 'frontier' not in belief:
            belief['frontier'] = set()
        if 'hidden_state' not in belief:
            belief['hidden_state'] = {}
        
        return belief

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action"""
        new_belief = {}
        for k, v in belief.items():
            if isinstance(v, list):
                new_belief[k] = v.copy()
            elif isinstance(v, set):
                new_belief[k] = v.copy()
            elif isinstance(v, dict):
                new_belief[k] = v.copy()
            else:
                new_belief[k] = v
        
        action_lower = action.lower().strip()
        
        if action_lower.startswith('turn left'):
            new_belief['orientation'] = (new_belief['orientation'] - 1) % 4
        elif action_lower.startswith('turn right'):
            new_belief['orientation'] = (new_belief['orientation'] + 1) % 4
        elif action_lower.startswith('move forward'):
            # Movement doesn't change object positions in this environment
            pass
        elif action_lower.startswith('pickup'):
            # Remove picked up object from scene
            obj_name_match = re.search(r'pickup ([^,]+)$', action_lower)
            if obj_name_match:
                obj_name = obj_name_match.group(1)
                new_belief['objects'] = [obj for obj in new_belief['objects'] if obj['full_name'] != obj_name]
                new_belief['carrying'] = obj_name
        elif action_lower.startswith('drop'):
            # Drop currently carried object in front of agent
            if new_belief['carrying']:
                # Check if there's already an object right in front
                front_objects = [obj for obj in new_belief['objects'] if obj['front_steps'] == 0 and obj['lateral_steps'] == 0]
                if not front_objects:  # Only drop if no object is already in front
                    # Add the dropped object to the front of the agent
                    dropped_obj = {
                        'name': new_belief['carrying'].split(' ')[0],  # Extract object type
                        'id': new_belief['carrying'].split(' ')[1] if len(new_belief['carrying'].split(' ')) > 1 else '1',
                        'full_name': new_belief['carrying'],
                        'front_steps': 0,  # Dropped right in front
                        'lateral_steps': 0,
                        'lateral_dir': 'center'
                    }
                    new_belief['objects'].append(dropped_obj)
                new_belief['carrying'] = None
        elif action_lower.startswith('go to'):
            # Navigation action - moves agent to object location
            obj_name_match = re.search(r'go to ([^,]+)$', action_lower)
            if obj_name_match:
                target_obj_name = obj_name_match.group(1)
                # Find target object
                target_obj = None
                for obj in new_belief['objects']:
                    if obj['full_name'] == target_obj_name:
                        target_obj = obj
                        break
                if target_obj:
                    # Agent moves to object position - object is now in front
                    new_belief['objects'] = [obj for obj in new_belief['objects'] if obj['full_name'] != target_obj_name]
                    target_obj['front_steps'] = 0
                    target_obj['lateral_steps'] = 0
                    target_obj['lateral_dir'] = 'center'
                    new_belief['objects'].append(target_obj)
        elif action_lower.startswith('toggle and go through') or action_lower.startswith('go through'):
            # Door traversal - removes door from scene
            door_match = re.search(r'(?:toggle and go through|go through) ([^,]+)$', action_lower)
            if door_match:
                door_name = door_match.group(1)
                new_belief['objects'] = [obj for obj in new_belief['objects'] if obj['full_name'] != door_name]
        
        return new_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate observation text from belief state"""
        # Handle check available actions specially
        if action.lower().strip() == "check available actions":
            return self._generate_available_actions_response(belief)
        
        lines = []
        
        if belief['goal']:
            lines.append(f"Your goal: {belief['goal']}")
            
        if belief['objects']:
            lines.append("In front of you in this room, you can see several objects:")
            for obj in belief['objects']:
                if obj['lateral_steps'] == 0 and (obj['lateral_dir'] == 'center' or obj['lateral_dir'] == ''):
                    if obj['front_steps'] == 0:
                        lines.append(f"There is a {obj['name']} {obj['id']} right in front of you.")
                    else:
                        lines.append(f"There is a {obj['name']} {obj['id']} {obj['front_steps']} steps in front of you.")
                else:
                    direction = obj['lateral_dir']
                    lines.append(f"There is a {obj['name']} {obj['id']} {obj['front_steps']} steps in front of you and {obj['lateral_steps']} steps to your {direction}.")
        else:
            lines.append("In front of you in this room, you can see several objects: The room has walls around you.")
            
        if belief['facing']:
            lines.append(f"You are facing {belief['facing']}.")
        else:
            lines.append("You are facing a wall.")
            
        if belief['carrying']:
            lines.append(f"You are carrying {belief['carrying']}.")
        else:
            lines.append("You are not carrying anything.")
            
        return " ".join(lines)

    def _generate_available_actions_response(self, belief) -> str:
        """Generate response for 'check available actions' command"""
        # This would normally extract available actions from the belief state
        # For now, we'll return a placeholder that indicates the action was processed
        return "You can take the following actions: turn left, turn right, move forward, check available actions"

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Extract valid action templates"""
        return {
            "go to": ["go to <object>"],
            "pickup": ["pickup <object>"],
            "drop": ["drop"],
            "toggle": ["toggle", "toggle and go through <door>"],
            "go through": ["go through <door>"],
            "move": ["move forward"],
            "turn left": ["turn left"],
            "turn right": ["turn right"],
            "check": ["check available actions"]
        }