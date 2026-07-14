# Prompt Engineering basic concepts

"Prompt Engineering" refers to the process of interacting efficiently with language models (LLMs).

## Main Objectives

- Maximize response efficiency

- Minimize ambiguities inherent to interaction

- Adjust the response/interaction to the problem context through the provision of extra information

## Strategies

- Be Specific when formulating a question

- Be Clear and Concise to avoid misunderstandings

- Include relevant context for response generation

- Use Examples to guide the model on the type of response expected

```
vague:    "Write about dogs."
specific: "Write a 3-sentence, upbeat product description for a durable leash
           aimed at owners of large, energetic breeds."
```

The specific version fixes length, tone, audience, and subject up front, leaving the model far less room
to guess wrong about what's actually wanted.

## Challenges

Each language model can interpret a prompt differently, so constant adjustments are necessary during the development process.

It is also necessary to balance between detail and flexibility, so that the model is capable of understanding what needs to be done while still having "creativity" to generate the final response.

