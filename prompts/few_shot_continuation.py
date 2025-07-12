class FewShotContinuation:
    def __init__(self):
        self.templates = {
            "example_caption": """I have some text samples, please help me complete one sample based on my example samples.


Example samples:
{examples}


Sample to complete:
{incomplete_prefix}""",
            "no_caption": """I have the following text samples, please help me complete the last sample based on the others:

{examples}

{incomplete_prefix}""",
            "paraphrase":"""Paraphrase one sample with the style of the example samples:

Example samples:

{examples}

Sample to paraphrase:

{incomplete_suffix}"""
        }

    def prepare_prompt(self, prompt_type, example_texts, incomplete_prefix, incomplete_suffix):
        template = self.templates.get(prompt_type, f"Invalid prompt type:{prompt_type}")
        examples = '\n\n'.join(example_texts)
        if prompt_type in ["example_caption", "no_caption"]:
            results = template.format(examples=examples, incomplete_prefix=incomplete_prefix)
        elif prompt_type in ["paraphrase"]:
            results = template.format(examples=examples, incomplete_suffix=incomplete_suffix)
        return results
    
if __name__ == '__main__':
    prompt_template = FewShotContinuation()
    print(prompt_template.prepare_prompt('example_caption', ['1', '2', '3'], '4'))
