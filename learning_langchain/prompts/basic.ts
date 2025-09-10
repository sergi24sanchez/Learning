import { PromptTemplate } from "@langchain/core/prompts";
/*
    In reality, you would get a user's input and then add it to the prompt
    before sending it to the large language model.
*/
export const run = async () => {
    const template = "What is the capital city of {country}?";
    const prompt = new PromptTemplate({
        inputVariables: ["country"],
        template: template, // template without external input
    });
    const res = await prompt.format({
        country: "France",
    });
    console.log(res);
};

run();