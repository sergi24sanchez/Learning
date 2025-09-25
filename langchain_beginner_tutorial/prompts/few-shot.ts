import { PromptTemplate, FewShotPromptTemplate } from "@langchain/core/prompts";
/*
    Few-shot prompt template is a prompt template that uses a few shot examples to generate a response.
*/
export const run = async () => {

    const examples = [
        {
            country: "United States",
            capital: "Washington, D.C.",
        },
        {
            country: "Germany",
            capital: "Berlin",
        },
    ];
    /*+ Next, provide the template to format the examples we have provided.
        We use the PrompTemplate class for this.*/
    const exampleFormattedTemplate = "Country: {country}\nCapital: {capital}\n";
    const examplePrompt = new PromptTemplate({
        inputVariables: ["country", "capital"],
        template: exampleFormattedTemplate, // template without external input
    });

    console.log("examplePrompt", await examplePrompt.format(examples[0]));

    const fewShotPrompt = new FewShotPromptTemplate({
        examples,
        examplePrompt,
        prefix: "What is the capital city of the following countries?",
        suffix: "Country: {country}\nCapital:",
        inputVariables: ["country"],
        exampleSeparator: "\n\n",
        templateFormat: "f-string",
    });
    const res = await fewShotPrompt.format({
        country: "France",
    });
    console.log("res",res);
};

run();