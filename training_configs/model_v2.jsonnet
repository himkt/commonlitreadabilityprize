{
    dataset_reader: {
        type: "commonlit_reader",
        tokenizer: {
            type: "pretrained_transformer",
            model_name: "../input/robertalarge",
        },
        excerpt_token_indexers: {
            tokens: {
                type: "pretrained_transformer",
                model_name: "../input/robertalarge",
            },
        },
    },
    train_data_path: "./processed_train.csv",
    validation_data_path: "./processed_valid.csv",
    model: {
        type: "naive",
        excerpt_embedder: {
            type: "basic",
            token_embedders: {
                tokens: {
                    type: "pretrained_transformer",
                    model_name: "../input/robertalarge",
                },
            },
        },
        excerpt_encoder: {
            type: "bert_pooler",
            pretrained_model: "../input/robertalarge",
        }
    },
    trainer: {
        num_epochs: 15,
        learning_rate_scheduler: {
            type: "slanted_triangular",
            num_epochs: 10,
            num_steps_per_epoch: 3088,
            cut_frac: 0.06
        },
        optimizer: {
            type: "huggingface_adamw",
            lr: 5e-7,
            weight_decay: 0.05,
        },
        validation_metric: "-loss"
    },
    data_loader: {
        batch_size: 8,
        shuffle: true
    }
}
