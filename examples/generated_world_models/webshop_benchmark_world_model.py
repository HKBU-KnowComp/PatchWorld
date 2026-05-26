from patchworld.worldmodel_base import BaseWorldModel
import re

class WebshopWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse the observation text into a structured dictionary."""
        parts = [part.strip() for part in obs_text.split(" [SEP] ")]
        obs_dict = {
            "type": "",
            "instruction": "",
            "navigation": [],
            "items": [],
            "product_info": {},
            "page_info": {}
        }
        
        i = 0
        if not parts:
            return obs_dict
            
        # Check if this is a results page or product page by looking for instruction first
        if parts[0] == "Instruction:" or (len(parts) > 1 and parts[1] == "Instruction:"):
            obs_dict["type"] = "instruction"
            # Find instruction
            instr_idx = parts.index("Instruction:") + 1 if "Instruction:" in parts else 1
            if instr_idx < len(parts):
                obs_dict["instruction"] = parts[instr_idx]
                i = instr_idx + 1
            
            page_number = 1
            while i < len(parts):
                if parts[i] == "Back to Search":
                    obs_dict["navigation"].append("Back to Search")
                    i += 1
                elif parts[i] == "< Prev":
                    obs_dict["navigation"].append("< Prev")
                    i += 1
                elif parts[i] == "Next >":
                    obs_dict["navigation"].append("Next >")
                    i += 1
                elif parts[i].startswith("Page "):
                    obs_dict["navigation"].append(parts[i])
                    # Extract page number
                    page_match = re.search(r"Page (\d+)", parts[i])
                    if page_match:
                        page_number = int(page_match.group(1))
                    i += 1
                elif parts[i] == "Search":
                    obs_dict["navigation"].append("Search")
                    i += 1
                elif re.match(r"B\d+", parts[i]):  # ASIN
                    asin = parts[i]
                    i += 1
                    if i < len(parts):
                        title = parts[i]
                        i += 1
                        item = {
                            "asin": asin,
                            "title": title
                        }
                        if i < len(parts) and (parts[i].startswith("$") or "to $" in parts[i]):
                            item["price"] = parts[i]
                            i += 1
                        obs_dict["items"].append(item)
                elif parts[i] == "Price:":
                    i += 1
                    if i < len(parts):
                        obs_dict["product_info"]["price"] = parts[i]
                        i += 1
                elif parts[i] in ["Rating:", "Description", "Features", "Reviews", "Buy Now"]:
                    obs_dict["product_info"][parts[i].lower().replace(" ", "_").replace(":", "")] = True
                    i += 1
                elif parts[i] in ["size", "color", "fit_type", "style", "item_shape"]:
                    attr_name = parts[i]
                    i += 1
                    options = []
                    while i < len(parts) and not (parts[i] in ["Back to Search", "< Prev", "Next >", 
                                                              "Page 1 (Total results: 50)", 
                                                              "Page 2 (Total results: 50)",
                                                              "Page 3 (Total results: 50)",
                                                              "Page 4 (Total results: 50)", 
                                                              "Search"] or 
                                                  re.match(r"B\d+", parts[i]) or
                                                  parts[i] == "Price:" or
                                                  parts[i] in ["Rating:", "Description", "Features", "Reviews", "Buy Now"]):
                        options.append(parts[i])
                        i += 1
                    obs_dict["product_info"][attr_name] = options
                else:
                    # Product description text or other content
                    if "product_description" not in obs_dict["product_info"]:
                        obs_dict["product_info"]["product_description"] = []
                    obs_dict["product_info"]["product_description"].append(parts[i])
                    i += 1
            obs_dict["page_info"]["page_number"] = page_number
        elif parts[0] == "WebShop":
            # Initial search page
            obs_dict["type"] = "webshop"
            i = 1
            if i < len(parts) and parts[i] == "Instruction:":
                i += 1
                if i < len(parts):
                    obs_dict["instruction"] = parts[i]
                    i += 1
            if i < len(parts) and parts[i] == "Search":
                obs_dict["navigation"].append("Search")
        elif parts[0] == "Thank you for shopping with us!":
            obs_dict["type"] = "purchase_complete"
            obs_dict["message"] = parts[0]
        
        return obs_dict

    def init_belief(self):
        """Initialize the belief state with latent support."""
        return {
            "page_type": "initial",
            "search_query": "",
            "instruction": "",
            "items": [],
            "selected_item": None,
            "product_attributes": {},
            "navigation_stack": [],
            # Adding latent belief fields as required by the environment
            "latent_variables": {
                "current_page": "initial",
                "selected_filters": {},
                "product_details": {},
                "search_history": [],
                "purchase_state": "not_started"
            },
            "facts": set(),
            "hypotheses": {},
            "frontier": [],
            "hidden_state": {}
        }

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct the belief state based on the observation."""
        obs = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        # Preserve latent belief structure
        if "latent_variables" not in belief:
            belief["latent_variables"] = {
                "current_page": "initial",
                "selected_filters": {},
                "product_details": {},
                "search_history": [],
                "purchase_state": "not_started"
            }
        if "facts" not in belief:
            belief["facts"] = set()
        if "hypotheses" not in belief:
            belief["hypotheses"] = {}
        if "frontier" not in belief:
            belief["frontier"] = []
        if "hidden_state" not in belief:
            belief["hidden_state"] = {}
        
        if obs["type"] == "webshop":
            belief["page_type"] = "initial"
            belief["instruction"] = obs["instruction"]
            belief["latent_variables"]["current_page"] = "initial"
            belief["items"] = []
            belief["selected_item"] = None
            belief["product_attributes"] = {}
        elif obs["type"] == "instruction":
            belief["instruction"] = obs["instruction"]
            if "Back to Search" in obs["navigation"] and len(obs["navigation"]) == 1 and not obs["items"] and not obs["product_info"]:
                # This is back to search from product page - should show results
                belief["page_type"] = "results"
                belief["latent_variables"]["current_page"] = "results"
                belief["items"] = []
            elif "Back to Search" in obs["navigation"] and obs["items"]:
                # This is a results page with Back to Search
                belief["page_type"] = "results"
                belief["items"] = obs["items"]
                belief["latent_variables"]["current_page"] = "results"
            elif any("Page" in nav for nav in obs["navigation"]):
                # This is a results page
                belief["page_type"] = "results"
                belief["items"] = obs["items"]
                belief["latent_variables"]["current_page"] = "results"
            elif "Search" in obs["navigation"] and not obs["items"] and not obs["product_info"]:
                # This is the initial page but reached via back navigation
                belief["page_type"] = "initial"
                belief["latent_variables"]["current_page"] = "initial"
                belief["items"] = []
                belief["selected_item"] = None
                belief["product_attributes"] = {}
            elif obs["product_info"] and not obs["items"]:
                # This is a product page
                belief["page_type"] = "product"
                belief["latent_variables"]["current_page"] = "product"
                belief["product_attributes"] = obs["product_info"]
                belief["latent_variables"]["product_details"] = obs["product_info"]
            else:
                # Default to results page when in doubt
                belief["page_type"] = "results"
                belief["items"] = obs["items"]
                belief["latent_variables"]["current_page"] = "results"
        elif obs["type"] == "purchase_complete":
            belief["page_type"] = "purchase_complete"
            belief["latent_variables"]["current_page"] = "purchase_complete"
            belief["latent_variables"]["purchase_state"] = "completed"
            
        return belief

    def predict_belief(self, belief, action: str):
        """Predict the next belief state given an action."""
        next_belief = belief.copy()
        
        # Ensure latent belief structure is maintained
        if "latent_variables" not in next_belief:
            next_belief["latent_variables"] = {
                "current_page": "initial",
                "selected_filters": {},
                "product_details": {},
                "search_history": [],
                "purchase_state": "not_started"
            }
        if "facts" not in next_belief:
            next_belief["facts"] = set()
        if "hypotheses" not in next_belief:
            next_belief["hypotheses"] = {}
        if "frontier" not in next_belief:
            next_belief["frontier"] = []
        if "hidden_state" not in next_belief:
            next_belief["hidden_state"] = {}
        
        if action.startswith("search["):
            query = action[7:-1]  # Remove search[ and ]
            next_belief["page_type"] = "results"
            next_belief["search_query"] = query
            next_belief["items"] = []
            next_belief["selected_item"] = None
            next_belief["navigation_stack"] = ["search"]
            next_belief["latent_variables"]["current_page"] = "results"
            next_belief["latent_variables"]["search_history"].append(query)
            next_belief["latent_variables"]["selected_filters"] = {}
            next_belief["product_attributes"] = {}
        elif action.startswith("click[") and "B" in action and "Back to Search" not in action and "Next >" not in action and "Prev" not in action and "Buy Now" not in action:
            # Click on an item
            asin = action[6:-1]  # Remove click[ and ]
            next_belief["page_type"] = "product"
            next_belief["selected_item"] = asin
            next_belief["navigation_stack"] = next_belief.get("navigation_stack", []) + ["item_click"]
            next_belief["latent_variables"]["current_page"] = "product"
            if "latent_variables" not in next_belief:
                next_belief["latent_variables"] = {}
            if "product_details" not in next_belief["latent_variables"]:
                next_belief["latent_variables"]["product_details"] = {}
            next_belief["latent_variables"]["product_details"]["asin"] = asin
        elif action == "click[Back to Search]" or action == "Back to Search":
            next_belief["page_type"] = "results"  # Show results page, not initial page
            next_belief["navigation_stack"] = []
            next_belief["latent_variables"]["current_page"] = "results"
            next_belief["items"] = []
            next_belief["selected_item"] = None
            next_belief["product_attributes"] = {}
        elif action == "click[< Prev]":
            if next_belief.get("navigation_stack"):
                prev_action = next_belief["navigation_stack"].pop()
                if prev_action == "item_click":
                    next_belief["page_type"] = "results"
                    next_belief["selected_item"] = None
                    next_belief["latent_variables"]["current_page"] = "results"
                elif prev_action == "next_page":
                    next_belief["page_type"] = "results"
                    next_belief["latent_variables"]["current_page"] = "results"
        elif action == "click[Next >]":
            next_belief["page_type"] = "results"
            next_belief["navigation_stack"] = next_belief.get("navigation_stack", []) + ["next_page"]
            next_belief["latent_variables"]["current_page"] = "results"
        elif action == "click[Buy Now]":
            next_belief["page_type"] = "purchase_complete"
            next_belief["navigation_stack"] = next_belief.get("navigation_stack", []) + ["buy_now"]
            next_belief["latent_variables"]["current_page"] = "purchase_complete"
            next_belief["latent_variables"]["purchase_state"] = "completed"
        elif action.startswith("click[") and any(opt in action.lower() for opt in ["size", "color", "fit_type", "style", "item_shape"]):
            # Clicking on an attribute option
            option = action[6:-1]  # Remove click[ and ]
            next_belief["navigation_stack"] = next_belief.get("navigation_stack", []) + [f"option_{option}"]
            # Update latent variables with selected filters
            if "color" in action.lower():
                if "selected_filters" not in next_belief["latent_variables"]:
                    next_belief["latent_variables"]["selected_filters"] = {}
                next_belief["latent_variables"]["selected_filters"]["color"] = option
            elif "size" in action.lower():
                if "selected_filters" not in next_belief["latent_variables"]:
                    next_belief["latent_variables"]["selected_filters"] = {}
                next_belief["latent_variables"]["selected_filters"]["size"] = option
            elif "style" in action.lower():
                if "selected_filters" not in next_belief["latent_variables"]:
                    next_belief["latent_variables"]["selected_filters"] = {}
                next_belief["latent_variables"]["selected_filters"]["style"] = option
            
        return next_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate an observation string from the belief state."""
        # Ensure belief has latent support
        if "latent_variables" not in belief:
            belief["latent_variables"] = {
                "current_page": belief.get("page_type", "initial"),
                "selected_filters": {},
                "product_details": {},
                "search_history": [],
                "purchase_state": "not_started"
            }
            
        if belief["page_type"] == "initial":
            return f"WebShop [SEP] Instruction: [SEP] {belief['instruction']} [SEP] Search"
        elif belief["page_type"] == "results":
            obs_parts = [f"Instruction: [SEP] {belief['instruction']} [SEP] Back to Search"]
            if belief["items"]:
                # Add page navigation
                page_nav = ["Page 1 (Total results: 50)"]
                if "next_page" in belief.get("navigation_stack", []):
                    page_nav.append("Next >")
                elif len(belief["navigation_stack"]) > 0 and belief["navigation_stack"][-1] == "next_page":
                    page_nav = ["Page 2 (Total results: 50)", "< Prev", "Next >"]
                elif action == "click[Next >]":
                    page_nav = ["Page 2 (Total results: 50)", "< Prev", "Next >"]
                elif action == "click[< Prev]":
                    page_nav = ["Page 1 (Total results: 50)", "Next >"]
                obs_parts.extend(page_nav)
                
                # Add items (limit to first 10 to match typical webshop behavior)
                for item in belief["items"][:10]:
                    if "price" in item:
                        obs_parts.append(f"{item['asin']} [SEP] {item['title']} [SEP] {item['price']}")
                    else:
                        obs_parts.append(f"{item['asin']} [SEP] {item['title']}")
            else:
                obs_parts.append("Search")
            return " [SEP] ".join(obs_parts)
        elif belief["page_type"] == "product":
            obs_parts = [f"Instruction: [SEP] {belief['instruction']} [SEP] Back to Search"]
            
            # Add navigation based on how we got here
            if belief.get("navigation_stack") and "item_click" in belief["navigation_stack"]:
                obs_parts.append("< Prev")
            
            # Add product attributes if available
            product_attrs = belief.get("product_attributes", {})
            for attr, values in product_attrs.items():
                if attr in ["size", "color", "fit_type", "style", "item_shape"]:
                    obs_parts.append(attr)
                    if isinstance(values, list):
                        obs_parts.extend(values)
            
            # Add product info
            if "price" in product_attrs:
                obs_parts.append(f"Price: {product_attrs['price']}")
            
            # Add standard product page elements
            obs_parts.extend(["Rating: N.A.", "Description", "Features", "Reviews", "Buy Now"])
            
            return " [SEP] ".join(obs_parts)
        elif belief["page_type"] == "purchase_complete":
            return "Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B09P8D2Q1Q [SEP] options [SEP] {} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 1.0 [SEP] Reward Details [SEP] None"
        else:
            # Fallback for unknown page types - try to reconstruct from available info
            if belief.get("instruction"):
                return f"Instruction: [SEP] {belief['instruction']} [SEP] Back to Search [SEP] Search"
            else:
                return "WebShop [SEP] Instruction: [SEP]  [SEP] Search"

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Extract valid action forms for this environment."""
        return {
            "search": ["search[<query>]"],
            "click": [
                "click[<asin>]",
                "click[Back to Search]",
                "click[< Prev]",
                "click[Next >]",
                "click[Buy Now]",
                "click[Description]",
                "click[Features]",
                "click[Reviews]",
                "click[<option>]",  # For size/color options
                "click[Search]"
            ]
        }