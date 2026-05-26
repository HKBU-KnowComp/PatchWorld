from patchworld.worldmodel_base import BaseWorldModel
from collections import defaultdict
import re

class TextcraftWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse observation text into structured data."""
        obs = {
            "inventory": {},
            "crafting_recipes": [],
            "goal": None,
            "message": None,
            "goal_item": None,
            "last_action_outcome": None
        }
        
        lines = obs_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Inventory:"):
                if "You are not carrying anything" in line:
                    obs["inventory"] = {}
                else:
                    # Parse inventory items like [bricks] (1) [brick] (6)
                    items = re.findall(r'$([^$]+)$$(\d+)', line)
                    for item, count in items:
                        obs["inventory"][item.strip()] = int(count)
            elif line.startswith("Crafting commands:"):
                continue
            elif line.startswith("craft "):
                obs["crafting_recipes"].append(line)
            elif line.startswith("Goal:"):
                goal_text = line.replace("Goal:", "").strip()
                obs["goal"] = goal_text
                # Extract the goal item (e.g., "minecraft:stone_pickaxe")
                goal_match = re.search(r'minecraft:([a-z_]+)', goal_text)
                if goal_match:
                    obs["goal_item"] = goal_match.group(1)
            elif line.startswith("Got "):
                obs["message"] = line
                obs["last_action_outcome"] = "got"
            elif line.startswith("Crafted "):
                obs["message"] = line
                obs["last_action_outcome"] = "crafted"
            elif line.startswith("Could not execute") or "Could not find" in line:
                obs["message"] = line
                if "find enough items" in line:
                    obs["last_action_outcome"] = "not_enough_items"
                elif "find a valid recipe" in line:
                    obs["last_action_outcome"] = "invalid_recipe"
                else:
                    obs["last_action_outcome"] = "not_found"
        return obs

    def init_belief(self):
        """Initialize belief state with latent support."""
        return {
            "inventory": defaultdict(int),
            "recipes": set(),
            "crafting_recipes": [],
            "goal_item": None,
            "last_action": None,
            "last_obs": None,
            "last_action_outcome": None,
            "latent_variables": {
                "obtainable_items": set(),
                "known_unavailable_items": set(),
                "recipe_graph": {}
            }
        }

    def correct_belief(self, belief_prior, obs_text: str):
        """Update belief state based on observation."""
        obs = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        # Update inventory
        if obs["inventory"]:
            belief["inventory"] = defaultdict(int, obs["inventory"])
        
        # Add new recipes
        for recipe in obs["crafting_recipes"]:
            belief["recipes"].add(recipe)
            belief["crafting_recipes"].append(recipe)
            
        # Update goal
        if obs["goal_item"]:
            belief["goal_item"] = obs["goal_item"]
            
        # Update outcome
        if obs["last_action_outcome"]:
            belief["last_action_outcome"] = obs["last_action_outcome"]
            
        # Update latent variables based on message
        if obs["message"]:
            if "Got " in obs["message"]:
                item_match = re.search(r'Got (\d+) ([a-z_ ]+)', obs["message"])
                if item_match:
                    _, item = item_match.groups()
                    belief["latent_variables"]["obtainable_items"].add(item)
            elif "Could not find" in obs["message"] and "enough items" not in obs["message"]:
                item_match = re.search(r'Could not find ([a-z_ ]+)', obs["message"])
                if item_match:
                    item = item_match.group(1)
                    belief["latent_variables"]["known_unavailable_items"].add(item)
        
        belief["last_obs"] = obs
        return belief

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action."""
        next_belief = {
            "inventory": belief["inventory"].copy(),
            "recipes": belief["recipes"].copy(),
            "crafting_recipes": belief["crafting_recipes"].copy(),
            "goal_item": belief["goal_item"],
            "last_action": action,
            "last_obs": belief["last_obs"],
            "last_action_outcome": None,
            "latent_variables": {
                "obtainable_items": belief["latent_variables"]["obtainable_items"].copy(),
                "known_unavailable_items": belief["latent_variables"]["known_unavailable_items"].copy(),
                "recipe_graph": belief["latent_variables"]["recipe_graph"].copy()
            }
        }
        
        action = action.strip()
        
        # Handle crafting actions
        craft_match = re.match(r'craft (\d+) ([^(]+) using (.+)', action)
        if craft_match:
            quantity, item, ingredients_str = craft_match.groups()
            quantity = int(quantity)
            item = item.strip()
            
            # Parse ingredients
            ingredients = {}
            parts = ingredients_str.split(', ')
            for part in parts:
                ing_match = re.search(r'(\d+) (.+)', part.strip())
                if ing_match:
                    ing_qty, ing_name = ing_match.groups()
                    ingredients[ing_name] = int(ing_qty)
            
            # Check if we have enough ingredients
            can_craft = True
            for ing_name, ing_qty in ingredients.items():
                if next_belief["inventory"][ing_name] < ing_qty:
                    can_craft = False
                    break
            
            # If we can craft, update inventory
            if can_craft:
                # Deduct ingredients
                for ing_name, ing_qty in ingredients.items():
                    next_belief["inventory"][ing_name] -= ing_qty
                
                # Add crafted item
                next_belief["inventory"][item] += quantity
                next_belief["last_action_outcome"] = "crafted"
                next_belief["latent_variables"]["obtainable_items"].add(item)
            else:
                next_belief["last_action_outcome"] = "not_enough_items"
                
        # Handle get actions
        get_match = re.match(r'get (\d+) (.+)', action)
        if get_match:
            quantity, item = get_match.groups()
            quantity = int(quantity)
            # Check if item is known to be unavailable
            if item in next_belief["latent_variables"]["known_unavailable_items"]:
                next_belief["last_action_outcome"] = "not_found"
            else:
                next_belief["inventory"][item] += quantity
                next_belief["last_action_outcome"] = "got"
                next_belief["latent_variables"]["obtainable_items"].add(item)
            
        # Handle inventory actions
        if action == "inventory":
            next_belief["last_action_outcome"] = "inventory"
            
        return next_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate observation text from belief state."""
        if belief["last_action_outcome"] == "got":
            get_match = re.match(r'get (\d+) (.+)', belief["last_action"])
            if get_match:
                quantity, item = get_match.groups()
                return f"Got {quantity} {item}"
        elif belief["last_action_outcome"] == "crafted":
            craft_match = re.match(r'craft (\d+) ([^(]+) using (.+)', belief["last_action"])
            if craft_match:
                quantity, item, _ = craft_match.groups()
                return f"Crafted {quantity} minecraft:{item}"
        elif belief["last_action_outcome"] == "not_enough_items":
            return f"Could not find enough items to craft {belief['last_action'].split('craft ')[1].split(' using')[0]}"
        elif belief["last_action_outcome"] == "not_found":
            get_match = re.match(r'get (\d+) (.+)', belief["last_action"])
            if get_match:
                _, item = get_match.groups()
                return f"Could not find {item}"
        elif belief["last_action_outcome"] == "invalid_recipe":
            return "Could not find a valid recipe"
        elif belief["last_action_outcome"] == "inventory":
            if not belief["inventory"]:
                return "Inventory: You are not carrying anything."
            else:
                items = []
                for item, count in belief["inventory"].items():
                    if count > 0:
                        items.append(f"[{item}] ({count})")
                return "Inventory: " + " ".join(items)
        
        # Default fallback
        if not belief["inventory"]:
            return "Inventory: You are not carrying anything."
        else:
            items = []
            for item, count in belief["inventory"].items():
                if count > 0:
                    items.append(f"[{item}] ({count})")
            return "Inventory: " + " ".join(items)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action templates."""
        return {
            "craft": ["craft <quantity> <item> using <ingredients>"],
            "get": ["get <quantity> <item>"],
            "inventory": ["inventory"],
            "i": ["I <statement>"],
            "im": ["I'm <statement>"]
        }