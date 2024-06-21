import openai
openai.api_key = ""




def summarize(self):
        messages = [{"role": "system", "content":
            "You are a intelligent assistant."}]

        message = "summarize and shorten this text:" + self.text
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-4o", messages=messages
            )
        reply = chat.choices[0].message.content
        self.displayText(reply)
        messages.append({"role": "assistant", "content": reply})