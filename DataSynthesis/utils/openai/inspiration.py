'''
@Po-Kai 2023/12
A more convenient and user-friendly multilingual translation tool implemented using the OpenAI API.
'''
from .cores.base import OpenAIGenerator


class OpenAIInspirationPrompt(object):
    
    def __init__(self):
        self.prompt = self._generate_prompt()
    
    def _generate_prompt(self):
        return {
            "system": 'I want you to play the role of a professional Chinese lyricist and tell an adapted story based on the lyrics.',
            "user": 'Please write a story about the creation of this song in Chinese. NOTE: The answer should be a story, not an analytical tone to the lyrics. Here are the lyrics to a song, without segmentation: "{lyrics}"'
        }
  
        return base_prompt

    def __call__(self, lyrics):
        return [
            {"role": "system", "content": self.prompt["system"]},
            {"role": "user", "content": self.prompt["user"].format(lyrics=lyrics)},
        ]


class OpenAInspiration(OpenAIGenerator):

    def _get_prompt_factory(self):
        return OpenAIInspirationPrompt()

    def generate(self, lyrics):
        # In order to be compatible with existing codes.
        return self(lyrics=lyrics)
