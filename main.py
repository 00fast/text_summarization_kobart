def main():
    # Training the model
    train_model()

    # Loading the trained weights
    weights_path = 'weights.ckpt'
    model_name = "digit82/kobart-summarization"

    # Create the summarizer
    summarizer = MySummarizer(model_name, weights_path)

    # Get text input from user
    example_text = input("Enter the text you want to summarize: ")

    # Generate and print summary
    summary = summarizer.generate_summary(example_text)
    print("Generated Summary:", summary)

if __name__ == '__main__':
    main()