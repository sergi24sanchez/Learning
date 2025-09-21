import { ChatOllama } from "@langchain/ollama";
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "@langchain/chains";

export const run = async () => {
    const model = new ChatOllama({temperature: 0.1, model: "llama3"});
    const template = "What is the capital city of {country}?";
    const prompt = new PromptTemplate({template, inputVariables: ["country"]});
    // Create a chain that takes the user input, format it and then sends to the LLM
    const chain = new LLMChain({llm: model, prompt: prompt});
    // run the chain by passing the user input
    const res = await chain.call({country: "France"});

    console.log("res", res);
};

run();