from __future__ import annotations

import json
from typing import Any


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


OPTIMIZER_DEFINITIONS_V2 = [
    {
        "id": "AdamW8bit",
        "name": "AdamW8bit",
        "display_name": "AdamW8bit",
        "category": "stable",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.0001,
        "default_unet_lr": 0.0001,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine", "linear", "constant_with_warmup"],
        "optimizer_args_schema": {},
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_steps_confidence": "medium",
        "description": "安定した標準Optimizer。最初の比較基準。",
        "risk_note": "安定した標準Optimizer。最初の比較基準。",
        "compatibility_notes": ["通常LR指定で扱います。"],
    },
    {
        "id": "PagedAdamW8bit",
        "name": "PagedAdamW8bit",
        "display_name": "PagedAdamW8bit",
        "category": "stable/memory_saving",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.0001,
        "default_unet_lr": 0.0001,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine", "linear", "constant_with_warmup"],
        "optimizer_args_schema": {},
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_steps_confidence": "medium",
        "description": "AdamW8bitに近い省メモリ候補。",
        "risk_note": "bitsandbytes環境依存に注意してください。",
        "compatibility_notes": ["AdamW8bitに近い目安です。"],
    },
    {
        "id": "Adafactor",
        "name": "Adafactor",
        "display_name": "Adafactor",
        "category": "memory_saving/advanced",
        "lr_semantics": "relative_step",
        "default_learning_rate": None,
        "default_unet_lr": None,
        "default_text_encoder_lr": 0,
        "default_scheduler": "adafactor",
        "allowed_schedulers": ["adafactor", "constant_with_warmup", "constant"],
        "optimizer_args_schema": {"relative_step": "bool", "scale_parameter": "bool", "warmup_init": "bool"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "省メモリ寄りのAdvanced optimizer。",
        "risk_note": "relative_step=Trueでは通常LRとは意味が異なります。",
        "compatibility_notes": ["Auto profileとFixed profileでLR意味が変わります。"],
    },
    {
        "id": "Lion",
        "name": "Lion",
        "display_name": "Lion",
        "category": "experimental",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.00005,
        "default_unet_lr": 0.00005,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "実験的Optimizer。AdamWに似たUIでも挙動は異なります。",
        "risk_note": "効きが強めに出ることがあります。",
        "compatibility_notes": ["balancedは5e-5を初期値にします。"],
    },
    {
        "id": "Lion8bit",
        "name": "Lion8bit",
        "display_name": "Lion8bit",
        "category": "experimental/memory_saving",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.00005,
        "default_unet_lr": 0.00005,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "Lionの8bit候補。",
        "risk_note": "bitsandbytes等の依存状況に注意。",
        "compatibility_notes": ["availabilityは環境検出に依存します。"],
    },
    {
        "id": "PagedLion8bit",
        "name": "PagedLion8bit",
        "display_name": "PagedLion8bit",
        "category": "experimental/memory_saving",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.00005,
        "default_unet_lr": 0.00005,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "Paged Lionの8bit候補。",
        "risk_note": "bitsandbytes等の依存状況に注意。",
        "compatibility_notes": ["availabilityは環境検出に依存します。"],
    },
    {
        "id": "DAdaptAdam",
        "name": "DAdaptAdam",
        "display_name": "DAdaptAdam",
        "category": "auto_lr/advanced",
        "lr_semantics": "auto_lr_multiplier",
        "default_learning_rate": 1.0,
        "default_unet_lr": 1.0,
        "default_text_encoder_lr": 0,
        "default_scheduler": "constant",
        "allowed_schedulers": ["constant"],
        "optimizer_args_schema": {"decouple": "bool", "weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "自動LR系のAdvanced optimizer。",
        "risk_note": "learning_rate=1.0は通常LRではなく自動推定LRへの倍率です。",
        "compatibility_notes": ["schedulerはconstantを推奨します。"],
    },
    {
        "id": "DAdaptLion",
        "name": "DAdaptLion",
        "display_name": "DAdaptLion",
        "category": "auto_lr/experimental",
        "lr_semantics": "auto_lr_multiplier",
        "default_learning_rate": 1.0,
        "default_unet_lr": 1.0,
        "default_text_encoder_lr": 0,
        "default_scheduler": "constant",
        "allowed_schedulers": ["constant"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "DAdapt + Lion系の実験的Optimizer。",
        "risk_note": "比較用。挙動差と固定化に注意してください。",
        "compatibility_notes": ["schedulerはconstantを推奨します。"],
    },
    {
        "id": "Prodigy",
        "name": "Prodigy",
        "display_name": "Prodigy",
        "category": "auto_lr/advanced",
        "lr_semantics": "auto_lr_multiplier",
        "default_learning_rate": 1.0,
        "default_unet_lr": 1.0,
        "default_text_encoder_lr": 0,
        "default_scheduler": "constant",
        "allowed_schedulers": ["constant"],
        "optimizer_args_schema": {"decouple": "bool", "weight_decay": "float", "d_coef": "float", "use_bias_correction": "bool"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "自動LR系。短期収束の比較候補。",
        "risk_note": "強く効くことがあるため過学習注意。",
        "compatibility_notes": ["schedulerはconstantを推奨します。"],
    },
    {
        "id": "Custom",
        "name": "Custom",
        "display_name": "Custom",
        "category": "custom",
        "lr_semantics": "custom",
        "default_learning_rate": None,
        "default_unet_lr": None,
        "default_text_encoder_lr": None,
        "default_scheduler": "custom",
        "allowed_schedulers": [],
        "optimizer_args_schema": {},
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "手動指定用。",
        "risk_note": "互換性チェック対象外の設定を含む可能性があります。",
        "compatibility_notes": ["Raw Args利用時はsd-scripts仕様を確認してください。"],
    },
]


OPTIMIZER_PROFILES_V2 = [
    ("adamw8bit_sdxl_balanced", "AdamW8bit", "SDXL", "balanced", "AdamW8bit SDXL Balanced", 0.0001, 0.0001, 0, "cosine", {}, 2500, 5000, 8000, "SDXL標準の比較基準。", "安定した初期値です。"),
    ("paged_adamw8bit_sdxl_memory", "PagedAdamW8bit", "SDXL", "memory_saving", "PagedAdamW8bit SDXL Memory", 0.0001, 0.0001, 0, "cosine", {}, 2500, 5000, 8000, "省メモリ寄りのAdamW候補。", "環境依存に注意。"),
    ("adafactor_sdxl_auto", "Adafactor", "SDXL", "auto_lr", "Adafactor SDXL Auto", None, None, 0, "adafactor", {"relative_step": True, "scale_parameter": True, "warmup_init": True}, 2500, 4000, 8000, "Adafactor relative_step運用。", "通常LRとは意味が異なります。"),
    ("adafactor_sdxl_fixed", "Adafactor", "SDXL", "advanced", "Adafactor SDXL Fixed", 0.0001, 0.0001, 0, "constant_with_warmup", {"relative_step": False, "scale_parameter": False, "warmup_init": False}, 2500, 5000, 8000, "Adafactor固定LR運用。", "AdamWとは挙動が異なります。"),
    ("lion_sdxl_balanced", "Lion", "SDXL", "balanced", "Lion SDXL Balanced", 0.00005, 0.00005, 0, "cosine", {"weight_decay": 0.01}, 2500, 4000, 8000, "Lionを弱めに試すbalanced profile。", "実験的です。"),
    ("dadapt_adam_sdxl_auto", "DAdaptAdam", "SDXL", "auto_lr", "DAdaptAdam SDXL Auto", 1.0, 1.0, 0, "constant", {"decouple": True, "weight_decay": 0.01}, 2500, 4000, 8000, "DAdaptAdam自動LR倍率運用。", "learning_rate=1.0は倍率です。"),
    ("dadapt_lion_sdxl_auto", "DAdaptLion", "SDXL", "experimental", "DAdaptLion SDXL Auto", 1.0, 1.0, 0, "constant", {"weight_decay": 0.01}, 2500, 4000, 8000, "DAdaptLion実験profile。", "実験的です。"),
    ("prodigy_sdxl_auto", "Prodigy", "SDXL", "auto_lr", "Prodigy SDXL Auto", 1.0, 1.0, 0, "constant", {"decouple": True, "weight_decay": 0.01, "d_coef": 1.0, "use_bias_correction": True}, 2500, 4000, 8000, "Prodigy自動LR運用。", "強く効くことがあります。"),
    ("prodigy_sdxl_soft", "Prodigy", "SDXL", "advanced", "Prodigy SDXL Soft", 1.0, 1.0, 0, "constant", {"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}, 2500, 4000, 8000, "Prodigyを弱めに試すprofile。", "固定化に注意してください。"),
]


NETWORK_TYPES = [
    ("standard_lora", "standard_lora", "Standard LoRA", "networks.lora", "standard_lora", "available", {"network_dim": "int", "network_alpha": "int"}, 32, 16, "標準LoRA。Phase 12.1の実行対象。", "安定した標準network type。"),
    ("locon", "locon", "LoCon", "locon", "locon", "planned", {}, 32, 16, "Phase 12.4以降で対応予定。", "Phase 12.1では実行不可。"),
    ("loha", "loha", "LoHa", "loha", "loha", "planned", {}, 32, 16, "Phase 12.4以降で対応予定。", "Phase 12.1では実行不可。"),
    ("lokr", "lokr", "LoKr", "lokr", "lokr", "planned", {}, 32, 16, "Phase 12.4以降で対応予定。", "Phase 12.1では実行不可。"),
    ("lycoris", "lycoris", "LyCORIS", "lycoris.kohya", "lycoris", "planned", {}, 32, 16, "Phase 12.4以降で対応予定。", "Phase 12.1では実行不可。"),
    ("custom", "custom", "Custom", "", "custom", "unsupported", {}, None, None, "手動定義用。", "互換性チェック対象外です。"),
]


TRAINING_PURPOSES = [
    ("character_face", "character_face", "Character Face", "顔LoRA向け。", ["face_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 10),
    ("character_full_body", "character_full_body", "Character Full Body", "全身・衣装を含むキャラクター向け。", ["full_body_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 20),
    ("style", "style", "Style", "画風LoRA向け。", ["style_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 30),
    ("costume", "costume", "Costume", "衣装LoRA向け。", ["costume_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 40),
    ("object", "object", "Object", "物体LoRA向け。", ["object_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 50),
    ("concept", "concept", "Concept", "概念LoRA向け。", ["dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 60),
    ("custom", "custom", "Custom", "完全カスタム用途。", [], None, None, {}, 999),
]


COMMON_SDXL_PARAMS = {
    "network_module": "networks.lora",
    "save_model_as": "safetensors",
    "save_every_n_epochs": 1,
    "sample_every_n_epochs": 1,
    "sample_sampler": "euler_a",
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "cache_latents": True,
    "gradient_checkpointing": True,
    "max_data_loader_n_workers": 2,
    "persistent_data_loader_workers": True,
    "resolution": [1024, 1024],
    "clip_skip": 2,
}


def sdxl_params(**overrides: Any) -> dict[str, Any]:
    params = dict(COMMON_SDXL_PARAMS)
    params.update(overrides)
    return params


TRAINING_RECIPES_V2 = [
    {
        "id": "sdxl_character_face_adamw8bit_smoke",
        "name": "sdxl_character_face_adamw8bit_smoke",
        "display_name": "SDXL Character Face / AdamW8bit Smoke",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "smoke",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=4, network_alpha=2, train_batch_size=1, repeats=1, max_train_steps=2, save_every_n_steps=1, sample_every_n_steps=1, resolution=[512, 512]),
        "basic_params": {"network_dim": 4, "network_alpha": 2, "train_batch_size": 1, "repeats": 1, "max_train_steps": 2},
        "advanced_params": {},
        "raw_args": {},
        "target": (1, 2, 5, 1),
        "expected_behavior": "実行経路のsmoke確認用。",
        "risk_note": "品質評価には使いません。",
        "sort_order": 10,
    },
    {
        "id": "sdxl_character_face_adamw8bit_pilot_3epoch",
        "name": "sdxl_character_face_adamw8bit_pilot_3epoch",
        "display_name": "SDXL Character Face / AdamW8bit Pilot 3 Epoch",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "pilot",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=2, max_train_epochs=3, min_snr_gamma=5),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 2, "max_train_epochs": 3},
        "advanced_params": {"min_snr_gamma": 5},
        "raw_args": {},
        "target": (100, 150, 300, 3),
        "expected_behavior": "短時間でloss推移とepoch差を見る。",
        "risk_note": "完成品質ではありません。",
        "sort_order": 20,
    },
    {
        "id": "sdxl_character_face_adamw8bit_standard_6epoch",
        "name": "sdxl_character_face_adamw8bit_standard_6epoch",
        "display_name": "SDXL Character Face / AdamW8bit Standard 6 Epoch",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "balanced",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=6, min_snr_gamma=5, no_metadata=True),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 6},
        "advanced_params": {"min_snr_gamma": 5, "no_metadata": True},
        "raw_args": {},
        "target": (2500, 5000, 8000, 6),
        "expected_behavior": "本番寄り標準候補。",
        "risk_note": "候補epochを確認してください。",
        "sort_order": 30,
    },
    {
        "id": "sdxl_character_face_adamw8bit_standard_10epoch",
        "name": "sdxl_character_face_adamw8bit_standard_10epoch",
        "display_name": "SDXL Character Face / AdamW8bit Standard 10 Epoch",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "balanced_long",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=10),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 10},
        "advanced_params": {},
        "raw_args": {},
        "target": (2500, 5000, 8000, 10),
        "expected_behavior": "標準10epoch。候補比較前提。",
        "risk_note": "長めなので保存間隔に注意してください。",
        "sort_order": 40,
    },
    {
        "id": "sdxl_character_face_adamw8bit_generalize",
        "name": "sdxl_character_face_adamw8bit_generalize",
        "display_name": "SDXL Character Face / AdamW8bit Generalize",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "generalize",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.00005, unet_lr=0.00005, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=16, network_alpha=8, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 16, "network_alpha": 8, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {},
        "raw_args": {},
        "target": (2500, 5000, 8000, 6),
        "expected_behavior": "弱めで固定化を避ける。",
        "risk_note": "顔再現が弱い場合があります。",
        "sort_order": 50,
    },
    {
        "id": "sdxl_character_face_lion_balanced_experimental",
        "name": "sdxl_character_face_lion_balanced_experimental",
        "display_name": "SDXL Character Face / Lion Balanced / Experimental",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "Lion",
        "optimizer_profile_id": "lion_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "experimental",
        "params": sdxl_params(optimizer_type="Lion", lr_scheduler="cosine", learning_rate=0.00005, unet_lr=0.00005, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"weight_decay": 0.01}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"weight_decay": 0.01}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "Lionを弱めに比較する。",
        "risk_note": "Experimental。実学習評価は未完了。",
        "sort_order": 100,
    },
    {
        "id": "sdxl_character_face_adafactor_auto_advanced",
        "name": "sdxl_character_face_adafactor_auto_advanced",
        "display_name": "SDXL Character Face / Adafactor Auto / Advanced",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "Adafactor",
        "optimizer_profile_id": "adafactor_sdxl_auto",
        "network_type_id": "standard_lora",
        "recipe_type": "advanced",
        "params": sdxl_params(optimizer_type="Adafactor", lr_scheduler="adafactor", text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"relative_step": True, "scale_parameter": True, "warmup_init": True}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"relative_step": True, "scale_parameter": True, "warmup_init": True}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "Adafactor Autoを比較する。",
        "risk_note": "Advanced。通常LRとは意味が異なります。",
        "sort_order": 110,
    },
    {
        "id": "sdxl_character_face_dadapt_adam_auto_advanced",
        "name": "sdxl_character_face_dadapt_adam_auto_advanced",
        "display_name": "SDXL Character Face / DAdaptAdam Auto / Advanced",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "DAdaptAdam",
        "optimizer_profile_id": "dadapt_adam_sdxl_auto",
        "network_type_id": "standard_lora",
        "recipe_type": "advanced",
        "params": sdxl_params(optimizer_type="DAdaptAdam", lr_scheduler="constant", learning_rate=1.0, unet_lr=1.0, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"decouple": True, "weight_decay": 0.01}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"decouple": True, "weight_decay": 0.01}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "DAdaptAdamを比較する。",
        "risk_note": "Advanced。learning_rate=1.0は倍率です。",
        "sort_order": 120,
    },
    {
        "id": "sdxl_character_face_prodigy_soft_advanced",
        "name": "sdxl_character_face_prodigy_soft_advanced",
        "display_name": "SDXL Character Face / Prodigy Soft / Advanced",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "Prodigy",
        "optimizer_profile_id": "prodigy_sdxl_soft",
        "network_type_id": "standard_lora",
        "recipe_type": "advanced",
        "params": sdxl_params(optimizer_type="Prodigy", lr_scheduler="constant", learning_rate=1.0, unet_lr=1.0, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "Prodigy Softを比較する。",
        "risk_note": "Advanced。強く効くことがあります。",
        "sort_order": 130,
    },
]


def optimizer_definition_v2_rows(now: str):
    for item in OPTIMIZER_DEFINITIONS_V2:
        yield (
            item["id"],
            item["name"],
            item["display_name"],
            item["category"],
            item["lr_semantics"],
            item["default_learning_rate"],
            item["default_unet_lr"],
            item["default_text_encoder_lr"],
            item["default_scheduler"],
            json_dumps(item["allowed_schedulers"]),
            json_dumps(item["optimizer_args_schema"]),
            item["target_steps_min"],
            item["target_steps_recommended"],
            item["target_steps_max"],
            item["target_steps_confidence"],
            item["description"],
            item["risk_note"],
            json_dumps(item["compatibility_notes"]),
            1,
            1,
            now,
            now,
        )


def optimizer_profile_v2_rows(now: str):
    for row in OPTIMIZER_PROFILES_V2:
        (
            profile_id,
            optimizer_definition_id,
            model_family,
            profile_type,
            display_name,
            learning_rate,
            unet_lr,
            text_encoder_lr,
            scheduler,
            args,
            target_min,
            target_recommended,
            target_max,
            description,
            risk_note,
        ) = row
        yield (
            profile_id,
            optimizer_definition_id,
            model_family,
            profile_type,
            display_name,
            learning_rate,
            unet_lr,
            text_encoder_lr,
            scheduler,
            json_dumps(args),
            target_min,
            target_recommended,
            target_max,
            description,
            risk_note,
            1,
            1,
            now,
            now,
        )


def network_type_rows(now: str):
    for row in NETWORK_TYPES:
        yield (*row[:6], json_dumps(row[6]), row[7], row[8], row[9], row[10], 1, 1, now, now)


def training_purpose_rows(now: str):
    for row in TRAINING_PURPOSES:
        purpose_id, name, display_name, description, roles, validation_preset, sample_template, overlay, sort_order = row
        yield (
            purpose_id,
            name,
            display_name,
            description,
            json_dumps(roles),
            validation_preset,
            sample_template,
            json_dumps(overlay),
            sort_order,
            1,
            1,
            now,
            now,
        )


def training_recipe_v2_rows(now: str):
    for recipe in TRAINING_RECIPES_V2:
        target_min, target_recommended, target_max, checkpoint_count = recipe["target"]
        yield (
            recipe["id"],
            recipe["name"],
            recipe["display_name"],
            recipe["model_family"],
            recipe["training_purpose_id"],
            recipe["optimizer_definition_id"],
            recipe["optimizer_profile_id"],
            recipe["network_type_id"],
            recipe["recipe_type"],
            json_dumps(recipe["params"]),
            json_dumps(recipe["basic_params"]),
            json_dumps(recipe["advanced_params"]),
            json_dumps(recipe["raw_args"]),
            json_dumps({"phase": "12.1", "network_type_must_be_available": True}),
            target_min,
            target_recommended,
            target_max,
            checkpoint_count,
            recipe["expected_behavior"],
            recipe["risk_note"],
            recipe["sort_order"],
            1,
            1,
            now,
            now,
        )


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def merge_params(*parts: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        for key, value in part.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                nested = dict(result[key])
                nested.update(value)
                result[key] = nested
            else:
                result[key] = value
    return result


def compatibility_check(params: dict[str, Any], *, network_type: dict[str, Any] | None = None, optimizer_definition: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    network_type = network_type or {}
    optimizer_definition = optimizer_definition or {}
    text_lr_values = [
        float(params.get(key) or 0)
        for key in ("text_encoder_lr", "text_encoder_lr1", "text_encoder_lr2")
        if params.get(key) not in (None, "")
    ]
    text_lr_active = any(value > 0 for value in text_lr_values)
    if params.get("cache_text_encoder_outputs") and text_lr_active:
        errors.append("cache_text_encoder_outputs=true かつ text_encoder_lr > 0 は併用できません。")
    if params.get("network_train_unet_only") and text_lr_active:
        errors.append("network_train_unet_only=true かつ text_encoder_lr > 0 は併用できません。")
    optimizer_type = str(params.get("optimizer_type") or optimizer_definition.get("id") or "")
    scheduler = str(params.get("lr_scheduler") or optimizer_definition.get("default_scheduler") or "")
    if optimizer_type in {"DAdaptAdam", "DAdaptLion", "Prodigy"}:
        if scheduler and scheduler != "constant":
            warnings.append(f"{optimizer_type} は lr_scheduler=constant を推奨します。")
        lr = params.get("learning_rate")
        try:
            lr_value = float(lr)
            if abs(lr_value - 1.0) > 0.5:
                warnings.append(f"{optimizer_type} の learning_rate は通常LRではなく倍率です。1.0から大きく外れています。")
        except (TypeError, ValueError):
            pass
    if optimizer_type == "Adafactor":
        args = params.get("optimizer_args") if isinstance(params.get("optimizer_args"), dict) else {}
        if args.get("relative_step") and params.get("learning_rate") not in (None, ""):
            warnings.append("Adafactor relative_step=Trueではlearning_rate指定の意味が通常と異なります。")
    if optimizer_type in {"AdamW8bit", "PagedAdamW8bit", "Lion", "Lion8bit", "PagedLion8bit"}:
        try:
            if float(params.get("learning_rate") or 0) > 0.0002:
                warnings.append(f"{optimizer_type} の learning_rate が高めです。")
        except (TypeError, ValueError):
            pass
    availability = network_type.get("availability")
    if availability and availability != "available":
        errors.append(f"Network type {network_type.get('display_name') or network_type.get('id')} は Phase 12.1 では {availability} のため実行できません。")
    return {"errors": errors, "warnings": warnings, "ok": not errors}


def recipe_v2_snapshot(recipe: Any, optimizer_definition: Any, optimizer_profile: Any, network_type: Any, purpose: Any) -> dict[str, Any]:
    return {
        "recipe": dict(recipe) if recipe else None,
        "optimizer_definition": dict(optimizer_definition) if optimizer_definition else None,
        "optimizer_profile": dict(optimizer_profile) if optimizer_profile else None,
        "network_type": dict(network_type) if network_type else None,
        "training_purpose": dict(purpose) if purpose else None,
    }
