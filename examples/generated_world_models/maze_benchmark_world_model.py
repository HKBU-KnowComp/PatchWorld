from patchworld.worldmodel_base import BaseWorldModel
import re

class MazeWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse observation text into structured data."""
        # Extract goal position
        goal_match = re.search(r"The goal is at position (\d+), (\d+)", obs_text)
        if not goal_match:
            return None
        goal_x, goal_y = int(goal_match.group(1)), int(goal_match.group(2))
        
        # Extract current position
        pos_match = re.search(r"Your current position is at position (\d+), (\d+)", obs_text)
        if not pos_match:
            return None
        pos_x, pos_y = int(pos_match.group(1)), int(pos_match.group(2))
        
        # Extract wall information
        walls = []
        if "wall above" in obs_text or "walls above" in obs_text or "above you" in obs_text:
            walls.append("up")
        if "wall below" in obs_text or "walls below" in obs_text or "below you" in obs_text:
            walls.append("down")
        if "wall to your left" in obs_text or "walls to your left" in obs_text or "to your left" in obs_text:
            walls.append("left")
        if "wall to your right" in obs_text or "walls to your right" in obs_text or "to your right" in obs_text:
            walls.append("right")
            
        return {
            "goal": (goal_x, goal_y),
            "position": (pos_x, pos_y),
            "walls": walls
        }

    def init_belief(self):
        """Initialize belief state."""
        return {
            "goal": None,
            "position": None,
            "walls": []
        }

    def correct_belief(self, belief_prior, obs_text: str):
        """Update belief state with new observation."""
        parsed = self.parse_observation(obs_text)
        if parsed is None:
            return belief_prior
            
        # Update belief with parsed information
        belief = belief_prior.copy()
        if parsed["goal"] is not None:
            belief["goal"] = parsed["goal"]
        if parsed["position"] is not None:
            belief["position"] = parsed["position"]
        if parsed["walls"] is not None:
            belief["walls"] = parsed["walls"]
            
        return belief

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action."""
        # Create a copy of current belief
        next_belief = belief.copy()
        
        # Parse action
        action = action.strip().lower()
        
        # Update position based on action if it's a valid move
        if next_belief["position"] is not None:
            x, y = next_belief["position"]
            
            # Apply movement (based on examples: right increases y, down increases x)
            # Fix: when moving down, x should increase; when moving up, x should decrease
            if action == "move up" and "up" not in next_belief["walls"]:
                x -= 1
            elif action == "move down" and "down" not in next_belief["walls"]:
                x += 1
            elif action == "move left" and "left" not in next_belief["walls"]:
                y -= 1
            elif action == "move right" and "right" not in next_belief["walls"]:
                y += 1
                
            next_belief["position"] = (x, y)
            
        return next_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Convert belief state back to observation text."""
        if belief["goal"] is None or belief["position"] is None:
            return ""
            
        goal_x, goal_y = belief["goal"]
        pos_x, pos_y = belief["position"]
        
        # Start building the observation text
        obs_text = f"The goal is at position {goal_x}, {goal_y}. Your current position is at position {pos_x}, {pos_y}."
        
        # Add wall information
        walls = belief["walls"]
        if walls:
            wall_descriptions = []
            # Maintain deterministic ordering
            if "up" in walls:
                wall_descriptions.append("above you")
            if "down" in walls:
                wall_descriptions.append("below you")
            if "left" in walls:
                wall_descriptions.append("to your left")
            if "right" in walls:
                wall_descriptions.append("to your right")
                
            if len(wall_descriptions) == 1:
                obs_text += f" There is a wall {wall_descriptions[0]}."
            else:
                obs_text += f" There are walls {', '.join(wall_descriptions)}."
        else:
            obs_text += "."
                
        return obs_text

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action forms for this domain."""
        return {
            "move <X>": ["move up", "move down", "move left", "move right"]
        }