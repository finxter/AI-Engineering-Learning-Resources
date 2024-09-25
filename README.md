# Prompt Engineering Guide

Prompt engineering is a relatively new discipline focused on developing and optimizing prompts to efficiently utilize language models (LMs) for a wide variety of applications and research topics. Mastering prompt engineering helps in understanding the capabilities and limitations of large language models (LLMs). Researchers leverage prompt engineering to enhance the performance of LLMs on diverse tasks such as question answering and arithmetic reasoning. Developers use prompt engineering to design robust and effective prompting techniques that interface with LLMs and other tools.

## Table of Contents

- [Guides](#guides)
  - [Introduction](#introduction)
  - [Techniques](#techniques)
  - [Applications](#applications)
  - [Prompt Hub](#prompt-hub)
  - [Models](#models)
  - [Risks and Misuses](#risks-and-misuses)
  - [Papers](#papers)
  - [Tools](#tools)
  - [Notebooks](#notebooks)
  - [Datasets](#datasets)
  - [Additional Readings](#additional-readings)
- [Learning Resources 📚](#learning-resources-📚)
  - [Neural Networks](#neural-networks)
  - [AI Engineering](#ai-engineering)
  - [LLM Bootcamps](#llm-bootcamps)
  - [Transformers](#transformers)
  - [Understanding ChatGPT](#understanding-chatgpt)
  - [Courses and Guides](#courses-and-guides)
- [LLMs](#llms)
  - [OpenAI LLMs](#openai-llms)
  - [Hugging Face](#hugging-face)
  - [Other Notable LLMs](#other-notable-llms)
- [Chat and Agents](#chat-and-agents)
  - [Chatbots](#chatbots)
  - [Open-Source Agents](#open-source-agents)
  - [Agent Frameworks](#agent-frameworks)
- [Development](#development)
  - [Frameworks and Libraries](#frameworks-and-libraries)
  - [Programming Tools](#programming-tools)
- [Tools](#tools)
  - [Document and Data Tools](#document-and-data-tools)
  - [UI and Interface Tools](#ui-and-interface-tools)
  - [AI Applications](#ai-applications)
- [Contributing](#contributing)
- [License](#license)

## Guides

### Introduction
- [Prompt Engineering - Introduction](https://www.promptingguide.ai/introduction)

### Techniques
- [Prompt Engineering - Techniques](https://www.promptingguide.ai/techniques)
  - [Zero-Shot Prompting](https://www.promptingguide.ai/techniques/zeroshot)
  - [Few-Shot Prompting](https://www.promptingguide.ai/techniques/fewshot)
  - [Chain-of-Thought Prompting](https://www.promptingguide.ai/techniques/cot)
  - [Self-Consistency](https://www.promptingguide.ai/techniques/consistency)
  - [Generate Knowledge Prompting](https://www.promptingguide.ai/techniques/knowledge)
  - [Prompt Chaining](https://www.promptingguide.ai/techniques/prompt_chaining)
  - [Tree of Thoughts (ToT)](https://www.promptingguide.ai/techniques/tot)
  - [Retrieval Augmented Generation](https://www.promptingguide.ai/techniques/rag)
  - [Automatic Reasoning and Tool-use (ART)](https://www.promptingguide.ai/techniques/art)
  - [Automatic Prompt Engineer](https://www.promptingguide.ai/techniques/ape)
  - [Active-Prompt](https://www.promptingguide.ai/techniques/activeprompt)
  - [Directional Stimulus Prompting](https://www.promptingguide.ai/techniques/dsp)
  - [Program-Aided Language Models](https://www.promptingguide.ai/techniques/pal)
  - [ReAct Prompting](https://www.promptingguide.ai/techniques/react)
  - [Multimodal CoT Prompting](https://www.promptingguide.ai/techniques/multimodalcot)
  - [Graph Prompting](https://www.promptingguide.ai/techniques/graph)

### Applications
- [Prompt Engineering - Applications](https://www.promptingguide.ai/applications)
  - [Function Calling](https://www.promptingguide.ai/applications/function_calling)
  - [Generating Data](https://www.promptingguide.ai/applications/generating)
  - [Generating Synthetic Dataset for RAG](https://www.promptingguide.ai/applications/synthetic_rag)
  - [Tackling Generated Datasets Diversity](https://www.promptingguide.ai/applications/generating_textbooks)
  - [Generating Code](https://www.promptingguide.ai/applications/coding)
  - [Graduate Job Classification Case Study](https://www.promptingguide.ai/applications/workplace_casestudy)

### Prompt Hub
- [Prompt Engineering - Prompt Hub](https://www.promptingguide.ai/prompts)
  - [Classification](https://www.promptingguide.ai/prompts/classification)
  - [Coding](https://www.promptingguide.ai/prompts/coding)
  - [Creativity](https://www.promptingguide.ai/prompts/creativity)
  - [Evaluation](https://www.promptingguide.ai/prompts/evaluation)
  - [Information Extraction](https://www.promptingguide.ai/prompts/information-extraction)
  - [Image Generation](https://www.promptingguide.ai/prompts/image-generation)
  - [Mathematics](https://www.promptingguide.ai/prompts/mathematics)
  - [Question Answering](https://www.promptingguide.ai/prompts/question-answering)
  - [Reasoning](https://www.promptingguide.ai/prompts/reasoning)
  - [Text Summarization](https://www.promptingguide.ai/prompts/text-summarization)
  - [Truthfulness](https://www.promptingguide.ai/prompts/truthfulness)
  - [Adversarial Prompting](https://www.promptingguide.ai/prompts/adversarial-prompting)

### Models
- [Prompt Engineering - Models](https://www.promptingguide.ai/models)
  - [ChatGPT](https://www.promptingguide.ai/models/chatgpt)
  - [Code Llama](https://www.promptingguide.ai/models/code-llama)
  - [Flan](https://www.promptingguide.ai/models/flan)
  - [Gemini](https://www.promptingguide.ai/models/gemini)
  - [GPT-4](https://www.promptingguide.ai/models/gpt-4)
  - [LLaMA](https://www.promptingguide.ai/models/llama)
  - [Mistral 7B](https://www.promptingguide.ai/models/mistral-7b)
  - [Mixtral](https://www.promptingguide.ai/models/mixtral)
  - [OLMo](https://www.promptingguide.ai/models/olmo)
  - [Phi-2](https://www.promptingguide.ai/models/phi-2)
  - [Model Collection](https://www.promptingguide.ai/models/collection)

### Risks and Misuses
- [Prompt Engineering - Risks and Misuses](https://www.promptingguide.ai/risks)
  - [Adversarial Prompting](https://www.promptingguide.ai/risks/adversarial)
  - [Factuality](https://www.promptingguide.ai/risks/factuality)
  - [Biases](https://www.promptingguide.ai/risks/biases)

### Papers
- [Prompt Engineering - Papers](https://www.promptingguide.ai/papers)
  - [Overviews](https://www.promptingguide.ai/papers#overviews)
  - [Approaches](https://www.promptingguide.ai/papers#approaches)
  - [Applications](https://www.promptingguide.ai/papers#applications)
  - [Collections](https://www.promptingguide.ai/papers#collections)

### Tools
- [Prompt Engineering - Tools](https://www.promptingguide.ai/tools)

### Notebooks
- [Prompt Engineering - Notebooks](https://www.promptingguide.ai/notebooks)

### Datasets
- [Prompt Engineering - Datasets](https://www.promptingguide.ai/datasets)

### Additional Readings
- [Prompt Engineering - Additional Readings](https://www.promptingguide.ai/readings)

## Learning Resources 📚

These resources were curated from various GitHub repositories and online sources to help you deepen your understanding of prompt engineering and related fields.

### Neural Networks
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)  
  From Andrej Karpathy, former Director of AI at Tesla and now at OpenAI.

### AI Engineering
- [AI Engineering Academy](https://academy.finxter.com)  
  The popular Finxter academy with 150k subscribed users, offering 50+ courses and downloadable certificates.

### LLM Bootcamps
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)  
  From the creators of the [Full Stack Deep Learning](https://fullstackdeeplearning.com/) course and book.

### Transformers
- [Transformers](https://www.youtube.com/watch?v=XfpMkf4rD6E)  
  Introduction to Transformers with Andrej Karpathy.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)  
  A great visual explanation of the Transformer architecture.

### Understanding ChatGPT
- [How ChatGPT Really Works](https://bootcamp.uxdesign.cc/how-chatgpt-really-works-explained-for-non-technical-people-71efb078a5c9)  
  An accessible explanation of how ChatGPT operates.

### Courses and Guides
- [AI Engineering Academy](https://academy.finxter.com)  
  Full Program to becoming an AI engineer, affordable with downloadable course certificates.
- [ChatGPT Prompt Engineering for Developers!](https://www.deeplearning.ai/)  
  A comprehensive course by OpenAI employees.
- [Learn Prompting](https://learnprompting.org/)  
  A text-based course focused on prompting techniques.
- [Prompt Engineering by Lilian Weng](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) ⭐  
  An in-depth post from Lilian Weng, Head of Applied AI Research at OpenAI.
- [MLOps Guide](https://github.com/Nyandwi/machine_learning_complete/blob/main/010_mlops/1_mlops_guide.md)  
  A guide on MLOps best practices.
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)  
  A comprehensive course on MLOps.
- [Prompt Engineering Repository](https://github.com/brexhq/prompt-engineering)  
  Covers the history, strategies, guidelines, and safety recommendations for working with LLMs.
- [Gandalf](https://gandalf.lakera.ai/)  
  A fun tool to learn about prompt injection.
- [Practical Deep Learning](https://course.fast.ai/)  
  A course for applying deep learning and machine learning to practical problems.
- [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) ⭐  
  From Andrej Karpathy, building GPT from scratch in code.
- [AI Canon](https://a16z.com/2023/05/25/ai-canon/) ⭐  
  A curated list of essential AI resources.
- [Generative AI Learning Path](https://www.cloudskillsboost.google/paths/118) ⭐  
  A learning path on Generative AI products and technologies from Google Cloud.
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)  
  Google's best practices in machine learning.
- [AI Companion App](https://github.com/a16z-infra/companion-app)  
  A tutorial stack to create and host AI companions accessible via browser or SMS.

## LLMs

### OpenAI LLMs
- [OpenAI LLMs](https://openai.com/product/gpt-4) ⭐  
  Access to the best LLMs, including GPT-4 and GPT-3.5, used in ChatGPT.

### Hugging Face
- [Hugging Face](https://huggingface.co/) ⭐  
  The leading open-source AI community offering models, datasets, and spaces.

### Other Notable LLMs
- [JARVIS](https://github.com/microsoft/JARVIS)  
  An interface to connect numerous AI models.
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)  
  Evaluate and rank open-source LLMs.
- [Guidance](https://github.com/microsoft/guidance) ⭐  
  Control modern language models more effectively.
- [TheBloke on HF](https://huggingface.co/TheBloke) ⭐  
  Compiles top open-source models in various formats.
- [DemoGPT](https://github.com/melih-unsal/DemoGPT)  
  Create 🦜️🔗 LangChain apps using prompts.
- [Llama2 Web UI](https://github.com/liltom-eth/llama2-webui)  
  Run Llama 2 with a Gradio web UI on GPU or CPU.

## Chat and Agents

### Chatbots
- [ChatGPT](https://chat.openai.com/) ⭐  
  The leading chatbot built on GPT-3.5 and GPT-4.

### Open-Source Agents
- [Open-Assistant](https://github.com/LAION-AI/Open-Assistant)  
  An open-source chat agent interacting with external sources.
- [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT)  
  An experimental open-source attempt to make GPT-4 fully autonomous.
- [LoopGPT](https://github.com/farizrahman4u/loopgpt) ⭐  
  A modular reimplementation of Auto-GPT.
- [ThinkGPT](https://github.com/jina-ai/thinkgpt)  
  Implements Chain of Thought reasoning for LLMs.

### Agent Frameworks
- [Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents) ⭐  
  Provides a natural language API on top of transformers with curated tools.
- [GPT-Engineer](https://github.com/AntonOsika/gpt-engineer)  
  Specify requirements, and the AI builds the project.
- [Khoj](https://github.com/khoj-ai/khoj)  
  An AI personal assistant for your digital brain.
- [Danswer](https://github.com/danswer-ai/danswer)  
  Open-source enterprise question-answering.
- [simpleaichat](https://github.com/minimaxir/simpleaichat)  
  Python package for interfacing with chat apps.
- [RealChar](https://github.com/Shaunwei/RealChar)  
  A realistic character chatbot.
- [MetaGPT](https://github.com/geekan/MetaGPT)  
  Multi-Agent Framework for generating PRDs, designs, tasks, and repositories.
- [ChatGPT AutoExpert](https://github.com/spdustin/ChatGPT-AutoExpert)  
  Custom instructions for ChatGPT and advanced data analysis.

## Development

### Frameworks and Libraries
- [LangChain](https://github.com/hwchase17/langchain) ⭐  
  Framework for developing applications powered by LLMs.
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)  
  SDK for integrating LLMs with conventional programming languages.
- [Langcorn](https://github.com/msoedov/langcorn)  
  API server for serving LangChain models and pipelines.

### Programming Tools
- [Pinecone](https://www.pinecone.io/)  
  Vector database for long-term memory with models.
- [Chroma](https://www.trychroma.com/)  
  Open-source alternative to Pinecone.
- [Plugandplai](https://github.com/edreisMD/plugnplai)  
  Simplify plugin integration into open-source LLMs.
- [GPTCache](https://github.com/zilliztech/GPTCache)  
  Caching for LLM responses to save costs.
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)  
  Examples and best practices for building with OpenAI.
- [How to Build an Agent with LangChain](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_tool-using_agent_with_Langchain.ipynb)  
  Jupyter notebook for building agents with LangChain.
- [Mojo](https://docs.modular.com/mojo/)  
  A programming language bridging research and production.
- [smol developer](https://github.com/smol-ai/developer) ⭐  
  Your own personal junior developer.
- [smol plugin](https://github.com/gmchad/smol-plugin)  
  Automatically generate OpenAI plugins from API specifications.
- [Kor](https://eyurtsev.github.io/kor/tutorial.html)  
  A thin wrapper for extracting structured data from LLMs.
- [tiktoken](https://github.com/openai/tiktoken)  
  Fast BPE tokenizer used with OpenAI's models.
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/gpt/function-calling)  
  Standardize LLM output.
- [Vercel AI SDK](https://github.com/vercel-labs/ai)  
  Build AI-powered applications with React, Svelte, and Vue.
- [Code Interpreter API](https://github.com/shroominic/codeinterpreter-api)  
  Open-source implementation of the ChatGPT Code Interpreter.
- [Open Interpreter](https://github.com/KillianLucas/open-interpreter/)  
  Locally running implementation of OpenAI's Code Interpreter.

## Tools

### Document and Data Tools
- [Vault AI](https://github.com/pashpashpash/vault-ai)  
  Upload documents and ask questions about their content.
- [privateGPT](https://github.com/imartinez/privateGPT) ⭐  
  Document Q&A using open-source LLMs.
- [Quivr](https://github.com/StanGirard/quivr) ⭐  
  Your Generative AI second brain for files and thoughts.
- [h2oGPT](https://github.com/h2oai/h2ogpt) ⭐  
  Similar to privateGPT with GPU inference support.
- [localGPT](https://github.com/PromtEngineer/localGPT)  
  Uses Vicuna-7b and InstructorEmbeddings with GPU/CPU support.
- [rag-stack](https://github.com/psychic-api/rag-stack)  
  Deploy a private ChatGPT alternative within your VPC.

### UI and Interface Tools
- [LangFlow](https://github.com/logspace-ai/langflow) ⭐  
  Visual prototyping and experimentation with LangChain.
- [Flowise](https://github.com/FlowiseAI/Flowise)  
  Similar to LangFlow but with LangChainJS.
- [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
  Browser interface for Stable Diffusion based on Gradio.
- [Unofficial OpenAI Status](https://openai-status.llm-utils.org/) ⭐  
  In-depth OpenAI status page.

### AI Applications
- [PentestGPT](https://github.com/GreyDGL/PentestGPT) 🕵️  
  GPT-powered penetration testing tool.
- [TypingMind](https://www.typingmind.com/) ⭐  
  Enhanced UI for ChatGPT.
- [Dify](https://github.com/langgenius/dify)  
  Create and operate AI-native apps based on OpenAI GPT models.
- [txtai](https://github.com/neuml/txtai)  
  Semantic search and workflows powered by language models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT](LICENSE)
