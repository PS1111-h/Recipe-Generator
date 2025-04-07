import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import ast

class RecipeGenerator:
    def __init__(self, recipe_path, api_key="sk-809cacc753674c8e912dcac8c1f784a1", api_base="https://api.deepseek.com"):
        # Load and prepare the data
        self.recipes_df = pd.read_csv(recipe_path) #get the data path
        self.vectorizer = TfidfVectorizer(stop_words='english') #removing the stop words, data preprocessing
        
        #Get api key and api base
        openai.api_key = api_key
        openai.api_base = api_base
        
        #Prepare the data for keyword matching
        self.prepare_data()
        
    def prepare_data(self):
        #Use a combined text string for a better keyword matching
        self.recipes_df['combined_text'] = self.recipes_df.apply(
            lambda x: f"{x['Name']} {x['RecipeCategory']} {x['Keywords']} {x['RecipeIngredientParts']}",
            axis=1
        )
        
        #Have all the ingredients and stored as strings of lists
        self.recipes_df['ingredients_list'] = self.recipes_df['RecipeIngredientParts'].apply(
            lambda x: self.parse_string_list(x)
        )
        
        #Have the instructions and stored as strings of lists
        self.recipes_df['instructions_list'] = self.recipes_df['RecipeInstructions'].apply(
            lambda x: self.parse_string_list(x)
        )
        
        # Use fir_transform to get a TF-IDF table and getting the weight of each term in our document
        #Store them in our self recipe vector for similarity check
        self.recipe_vectors = self.vectorizer.fit_transform(self.recipes_df['combined_text'])
    
    def parse_string_list(self, string_list):
        if not isinstance(string_list, str):#Check if the input is a string
            return []
            
        try:
            #Check if there are c("item1", "item2") format in our dataset
            if string_list.startswith('c(') and string_list.endswith(')'):
                #Extract items between c( and )
                items_str = string_list[2:-1]
                items = []
                current_item = ""
                in_quotes = False
                
                #for every character in our item string, check if it is quoted by "". when we meet our first ", add that character into
                #our item list and until it meets the second "
                for char in items_str:
                    if char == '"' and (len(current_item) == 0 or current_item[-1] != '\\'):
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        if current_item.startswith('"') and current_item.endswith('"'):
                            current_item = current_item[1:-1]
                        items.append(current_item.strip())
                        current_item = ""
                    else:
                        current_item += char
                
                # Add the last item
                if current_item:
                    if current_item.startswith('"') and current_item.endswith('"'):
                        current_item = current_item[1:-1]
                    items.append(current_item.strip())
                
                return items
            else:
                #Convert literal format into list
                return ast.literal_eval(string_list)
        except:
            #If parsing fails, split by comma as fallback
            if ',' in string_list:
                return [item.strip() for item in string_list.split(',')]
            return [string_list]
    
    def find_recipes_by_keywords(self, keywords, top_n=10):#Takes in keywords and finding the top 10 relevent results
        #Join all the keywords to a single string
        query = ' '.join(keywords)
        
        #Convert the query string into a TF-IDF vector. 
        query_vector = self.vectorizer.transform([query])
        
        #Use cosine similarity how similar the query to every recipe in the datset
        similarities = cosine_similarity(query_vector, self.recipe_vectors).flatten()
        
        #Get top matches
        top_indices = similarities.argsort()[-top_n:][::-1]
        return self.recipes_df.iloc[top_indices]
    
    def generate_recipe_with_deepseek(self, keywords, dietary_restrictions=None, model="deepseek-chat"):
        #Find similar recipes for reference
        similar_recipes = self.find_recipes_by_keywords(keywords)
        
        #Extract a sample of ingredients from similar recipes
        sample_ingredients = []
        for _, recipe in similar_recipes.iterrows():
            if hasattr(recipe, 'ingredients_list') and isinstance(recipe.ingredients_list, list):
                sample_ingredients.extend(recipe.ingredients_list[:5])  # Take up to 5 ingredients from each
        
        #Remove duplicates
        sample_ingredients = list(set(sample_ingredients))[:10]  # Limit to 10 ingredients
        
        #Build the prompt
        prompt = f"Create a recipe using these keywords: {', '.join(keywords)}.\n"
        prompt += f"Some ingredients you might consider: {', '.join(sample_ingredients)}\n"
        
        if dietary_restrictions:
            prompt += f"The recipe should be suitable for {dietary_restrictions} diets.\n"
            
        prompt += "Format the recipe with a title, ingredients list with measurements, and step-by-step instructions."
        
        try:
            # Make the API request with deepseek
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a chef creating delicious recipes."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            # Extract the generated recipe text
            recipe_text = response['choices'][0]['message']['content']
            
        except Exception as e:
            #If there is an API error
            print(f"API error: {str(e)}")
            return {
                "title": f"Error generating recipe for {', '.join(keywords)}",
                "ingredients": ["API Error"],
                "instructions": [f"Could not generate recipe: {str(e)}"]
            }
        return self.parse_recipe_text(recipe_text)
    
    def parse_recipe_text(self, recipe_text):
        lines = recipe_text.split('\n')#divides the recipe tedxt into sparate lines for processing
        title = lines[0].strip('#').strip()#remove "#" to get our recipe title
        
        ingredients = []#Initialize a list for our ingredients
        instructions = []#Initialize a list for our ingredients
        
        section = None
        for line in lines[1:]:
            line = line.strip()#Clean each line
            if not line:
                continue
            #Detect which section we are in
            if "ingredient" in line.lower():
                section = "ingredients"
                continue
            elif "instruction" in line.lower():
                section = "instructions"
                continue
            
            if section == "ingredients" and (line.startswith('-') or any(c.isdigit() for c in line[:3])):
                ingredients.append(line.lstrip('- '))#Collect our ingredients need
            elif section == "instructions" and (line.startswith('-') or any(c.isdigit() for c in line[:3])):
                instructions.append(line.lstrip('- ').lstrip('1234567890. '))#Get our intructions
        
        return {#Return title ingredients and instrcutions as a dictionary
            "title": title,
            "ingredients": ingredients,
            "instructions": instructions
        }
    
    def generate_recipe(self, keywords, dietary_restrictions=None, model="deepseek-chat"):
        return self.generate_recipe_with_deepseek(keywords, dietary_restrictions, model)

if __name__ == "__main__":
    generator = RecipeGenerator(
        recipe_path="recipes.csv",
        api_key="sk-809cacc753674c8e912dcac8c1f784a1", 
        api_base="https://api.deepseek.com"  
    )
    
    # User input
    print("----Recipe Generator----")
    print("Hi, I am your personal chef. ")
    print("I would be happy to provide a a recipe based on your ingredients and preferences: ")

    #Get keywords from user
    keywords_input = input("Please enter ingredients(separated by commas): ")
    keywords = [k.strip() for k in keywords_input.split(",")]
    
    #Get dietary restrcition
    dietary_input = input("Any dietary restrictions? (for example, vegetarian, gluten-free, etc.) -- Press Enter to skip: ")
    dietary_restrictions = dietary_input.strip() if dietary_input.strip() else None
    
    print("\nGenerating your recipe...")
    
    #Generate the recipe
    recipe = generator.generate_recipe(
        keywords=keywords,
        dietary_restrictions=dietary_restrictions,
        model="deepseek-chat" #Using deepseek model to generate our receipt
    )
    
    #Print the recipe
    print("\n" + "-" * 50)
    print(f"# {recipe['title']}")#getting a creative title for the dish
    print("-" * 50 + "\n")
    
    print("## Ingredients\n")
    for ingredient in recipe['ingredients']:
        print(f"- {ingredient}")
    
    print("\n## Instructions\n")
    for i, step in enumerate(recipe['instructions']):
        print(f"{i+1}. {step}")
    
    print("\nEnjoy!")
    