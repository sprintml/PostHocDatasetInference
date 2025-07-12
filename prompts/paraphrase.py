class Paraphrase:
    def __init__(self):
        self.templates = {
            "keep_length": """Rewrite this entire text (all sentences with no exception) expressing the same meaning using very different words. 
Try your best to keep the style of the paraphrased text similar to that of the original text.
Aim to keep the rewriting similar in length to the original text. 
Do it three times, and make sure every rewritten text has very different words.
The text to be rewritten is identified as <Example 1>. 
Format your output as: 
Example 2: <insert paraphrase 2> 

Example 3: <insert paraphrase 3> 

Example 4: <insert paraphrase 4> 

Example 1: {original_text}
""",
        }

    def prepare_prompt(self, prompt_type, original_text):
        template = self.templates.get(prompt_type, f"Invalid prompt type:{prompt_type}")
        return template.format(original_text=original_text)
