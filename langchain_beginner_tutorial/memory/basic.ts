import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { ChatOllama } from "@langchain/ollama";

/**
 * Gives the chain the ability to remember information from previous interactions.
 * This is useful for chatbots and conversation bots.
 * 
 * `ConversationChain` is a simple type of memory that remembers all previous interactions.
 * and adds them as context that is passed into the LLM.
 */

export const run = async () => {
    const model = new ChatOllama({ model: "llama3" });

    const memory = new BufferMemory();
    const chain = new ConversationChain({ llm: model, memory });

    const firstResponse = await chain.invoke({ input: "Hello, my name is Sergi." });
    console.log(firstResponse);

    const secondResponse = await chain.invoke({ input: "What is my name?" });
    console.log(secondResponse);
};

run();
