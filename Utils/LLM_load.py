from openai import OpenAI
from Utils.config_setup import Config


setup = Config()

class llm: 
    def __init__ (
            self,
            user,
            system  ) :
        self.user = user
        self.system = system


    def llm_call(self):
        client = OpenAI(
            api_key = setup.api_key,
            base_url = "https://api.deepseek.com"
        )
    
        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [ {"role": "system", "content": self.system},
                         {"role": "user", "content":self.user}
                        
                    ],
            response_format={
                'type': 'json_object'
            }
        )
        output = response.choices[0].message.content
        return output
