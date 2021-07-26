{
    dataset_reader: {
        type: "commonlit_reader",
        tokenizer: {
            type: "pretrained_transformer",
            model_name: "../input/roberta-base",
        },
        token_indexers: {
            tokens: {
                type: "pretrained_transformer",
                model_name: "../input/roberta-base",
            },
        },
    },
    train_data_path: "./processed_train.csv",
    validation_data_path: "./processed_valid.csv",
    model: {
        type: "baseline",
        excerpt_embedder: {
            type: "basic",
            token_embedders: {
                tokens: {
                    type: "pretrained_transformer",
                    model_name: "../input/roberta-base",
                },
            },
        },
        excerpt_encoder: {
            type: "bert_pooler",
            pretrained_model: "../input/roberta-base",
        }
    },
    trainer: {
        num_epochs: 30,
        learning_rate_scheduler: {
            type: "slanted_triangular",
            num_epochs: 10,
            num_steps_per_epoch: 3088,
            cut_frac: 0.06
        },
        optimizer: {
            type: "huggingface_adamw",
            lr: 2e-5,
            weight_decay: 0.1,
        },
        validation_metric: "-loss"
    },
    data_loader: {
        batch_size: 16,
        shuffle: true
    }
}
