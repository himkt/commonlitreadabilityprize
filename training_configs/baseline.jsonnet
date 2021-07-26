{
    dataset_reader: {
        type: "commonlit_reader",
        tokenizer: "whitespace",
    },
    vocabulary: {
        "pretrained_files": {
            "tokens": "./glove.6B.100d.txt",
        },
    },
    train_data_path: "data/processed_train.csv",
    validation_data_path: "data/processed_valid.csv",
    model: {
        type: "baseline",
        excerpt_embedder: {
            type: "basic",
            token_embedders: {
                tokens: {
                    embedding_dim: 100,
                    pretrained_file: "./glove.6B.100d.txt"
                }
            }
        },
        excerpt_encoder: {
            type: "lstm",
            input_size: 100,
            hidden_size: 50,
        }
    },
    trainer: {
        num_epochs: 30,
        optimizer: {
            type: "adam",
            lr: 1e-5
        },
        validation_metric: "-loss"
    },
    data_loader: {
        batch_size: 16,
        shuffle: true
    }
}
