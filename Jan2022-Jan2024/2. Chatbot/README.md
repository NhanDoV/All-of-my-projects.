This project is a copy (also with fake-data) from the internal projects that I am joined, they included 3 versions
- [Rule-based chatbot](https://github.com/NhanDoV/All-of-my-projects./tree/main/Jan2022-Jan2024/2.%20Chatbot/Rule-based-chatbot)
- [Retrieval based](https://github.com/NhanDoV/All-of-my-projects./tree/main/Jan2022-Jan2024/2.%20Chatbot/Retrieval-based-bot)
- Chatbot used transformers: [demo notebook](https://github.com/NhanDoV/All-of-my-projects./blob/main/Jan2022-Jan2024/2.%20Chatbot/linh-tinh/transformers_model.ipynb)
- Chatbot using API from chatGPT

# 1. Rule-based chatbot
Rule-based chatbots are structured as a dialog tree and often use regular expressions to match a user’s input to human-like responses. 

The aim is to simulate the back-and-forth of a real-life conversation, often in a specific context, like 
- telling the user what the weather is like outside. 
- answering the users's question about the service

In chatbot design, rule-based chatbots are **closed-domain**, also called dialog agents, because they are limited to conversations on a specific subject.

![image](https://user-images.githubusercontent.com/60571509/233780873-939e104f-c40b-4617-b01a-fd6f05dbee9a.png)

### 1.1. Chatbot Intents
In chatbots design, an `intent` is the purpose or category of the user query. The user’s utterance gets matched to a chatbot intent. In rule-based chatbots, you can use regular expressions to match a user’s statement to a chatbot intent.

### 1.2. Chatbot Utterances
An `utterance` is a statement that the user makes to the chatbot. The chatbot attempts to match the utterance to an intent.

### 1.3. Chatbot Entities
An `entity` is a value that is parsed from a user utterance and passed for use within the user response.

## 2. Retrieval-based
Retrieval-based chatbots are used in closed-domain scenarios and rely on a `collection of predefined responses` to a user message. A retrieval-based bot completes three main tasks: intent classification, entity recognition, and response selection.

### 2.1. Intent Similarity for Retrieval-Based Chatbots
For retrieval-based chatbots, it is common to use `bag-of-words` or `tf-idf` to compute intent similarity.

### 2.2. Entity Recognition for Retrieval-Based Chatbots
For retrieval-based chatbots, entity recognition can be accomplished using `part-of-speech` (POS) tagging or word embeddings such as `word2vec`.

## 3. AI Chatbot
### 3.1. Transformers model 
[Reference](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html)

Transformer, proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762), is a neural network architecture solely based on self-attention mechanism and is very parallelizable.

![image](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjubJSdXkMwulhxB3g5tOEJ8Iihc7BMxsdEtJicdRWon1GZi8mhkpN-gN8heS8ZyJT4R7JZ_mLm_gqorKAvETrAKq1P3Msn7x9M7gU2iPkl0BBKevmuiyjMJRu3u186jem5yXEdIJ5mC1I/s1600/transformer.png)

Rather than `RNNs` or `CNNs`, a Transformer model manages variable-sized input by stacking self-attention layers. This generic architecture has several advantages:
- It makes no assumptions about the temporal or spatial relationships in the data. This is ideal for processing a collection of things.
- Layer outputs can be calculated in simultaneously, rather than in serial, as in an `RNN`.
- Distant elements can influence each other's output without going through many recurrent stages or convolution layers.
- It can learn long-term dependencies.
  
One downside of this architecture is that the output for a time-step is calculated based on the complete history, rather than only the inputs and present `hidden-state`. This may be less efficient.
- If the input has a temporal/spatial link, such as text, some positional encoding is required; otherwise, the model will see a bag of words.
- If you want to learn more about Transformers, check out [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and [Illustrated `Transformer`](http://jalammar.github.io/illustrated-transformer/).

  ![image](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi6xvc6UV_8MG2P6-j9kBkD-DL1c21U7DpLhFIwEOW3gOoKEgBFjsg0jo12yjga8sAC-GScl42Tby-cFdW7zu5KW9cgG_2LbwJatcNPOX2kF4t2e6xR19mmlD3S07MmPaDYVhe37MTj0hc/s1600/transformer.png)

[read-more](https://www.tensorflow.org/text/tutorials/transformer)

### 3.2. GPT models
As we know, a `GPT (Generative Pretrained Transformer)` chatbot is a type of AI chatbot that utilizes `GPT` models. `GPT` is a text generation model developed by `OpenAI`.

`Conversational AI platforms` are commonly employed for developing an `AI chatbot`. Examples of similar frameworks include `Google Dialogflow`, `Rasa`, and `Microsoft's Luis`. These platforms include functionality for managing conversations, interpreting user inquiries, identifying intents, and responding appropriately. The chatbot designer explains:

- Intents likes "want_apple"
- Sample words for each intent, such as "I want to buy an apple"
- Responses to each intent, such as "here is the list of our apples"

Essentially, this allows the designer to exactly determine the chatbot's response after determining the sentence's intent.

Finally, AI chatbots can use text generating models such as GPT. In this situation, unlike the previously shown AI chatbot, the designer is not required to construct responses. Instead, the model generates the responses automatically, however there is a danger of erroneous results.

There are several more or less free and open-source text generation models available. `ChatGPT` utilizes `OpenAI's Davinci GPT3-5` model. Their most recent model is `GPT-4`. `Google` offers `Bert`, while Meta provides OPT and recently announced `LLaMA`, an open-source model. 
They may use text generation models to automatically generate responses.

#### How to create your gpt chatbot with a chatbot platform?
- **Step 1: Prepare training data**. You will need a chat dataset to train your GPT model. Data can be obtained through live conversations, forums, social media, documentation, and other sources.
- **Step 2: Train and Configure the Platform**. Depending on the platform you've chosen, there might be specific instructions to follow to configure the GPT model.
- **Step 3: Deploy the Chatbot**. Once your model is trained, you can deploy it on the chatbot platform you've chosen. You can also integrate your chatbot with other systems or messaging channels, such as Facebook Messenger, Slack, or Teams.

#### How to optimize this
- **Optimize Your `Chatbot's UX`**. It is critical to ensure that your chatbot is easy to use and responds quickly and accurately. You may improve your chatbot's UX by including interesting welcome messages, clear menu options, and error messages to guide users. Also, inspect your chatbot's ergonomics.
- **Analyze User Data**. This helps you understand how users interact with your chatbot and find areas for improvement. You may use this information to fine-tune your chatbot and enhance the user experience.
- **Improve Your Chatbot's Learning**. You must add new data on a constant basis. You may increase your chatbot's accuracy by adding fresh data, adjusting model parameters, and using machine learning techniques.
- **Enable Human Takeover**. Although GPT chatbots can produce natural and correct responses, there may be times when a user need a human response. In these cases, you can allow users to contact a support representative to resolve their issue or answer their concerns. Improving your GPT chatbot is an ongoing endeavor. You may give an optimal user experience and increase the accuracy of your chatbot over time by refining its UX, analyzing user data, improving its learning, and giving a human contact option.
