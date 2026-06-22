import json

COMMON_PARAMS = {
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
}

PRESET_DEFS = [
    ("integration_smoke_sdxl", "Integration Smoke - SDXL", "SDXL", "sdxl_train_network.py", "実モデルで学習実行、出力LoRA、サンプル画像、DB取り込みの経路だけを短時間で確認する設定。", 1, 999, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":4,"network_alpha":2,"train_batch_size":1,"repeats":1,"max_train_steps":2,"save_every_n_steps":1,"sample_every_n_steps":1,"resolution":[512,512]}, "品質評価ではなくintegration smoke用。短時間で完走と成果物生成を確認する。", "LoRA品質の判断には使わない。"),
    ("sdxl_2d_face_pilot_3epoch", "SDXL 2D Face - Pilot 3 Epoch", "SDXL", "sdxl_train_network.py", "実用LoRA評価前の短時間テスト。品質完成ではなく、loss推移・epoch差・sample比較の確認用。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"cosine","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":2,"max_train_epochs":3,"resolution":"1024,1024","min_snr_gamma":5,"clip_skip":2,"save_every_n_epochs":1,"sample_every_n_epochs":1}, "50枚前後のデータセットで、各epochの差とloss推移を短時間で確認する。", "完成品質ではない。顔再現が弱い可能性があるが、まずは評価導線の確認を優先する。"),
    ("sdxl_2d_face_pilot_generalize_3epoch", "SDXL 2D Face - Pilot Generalize 3 Epoch", "SDXL", "sdxl_train_network.py", "固定化・過学習を避ける弱めの短時間テスト。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"cosine","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":16,"network_alpha":8,"train_batch_size":1,"repeats":2,"max_train_epochs":3,"resolution":"1024,1024","min_snr_gamma":5,"clip_skip":2,"save_every_n_epochs":1,"sample_every_n_epochs":1}, "50枚前後のデータセットで、弱め設定のepoch差とloss推移を短時間確認する。", "完成品質ではない。固定化を避けるぶん顔再現が弱い可能性がある。"),
    ("sdxl_2d_face_standard_6epoch", "SDXL 2D Face - Standard 6 Epoch", "SDXL", "sdxl_train_network.py", "Dataset整備後の本番寄り短時間学習。顔特徴の入り、epoch別変化、過学習傾向を見る。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"cosine","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":2,"max_train_epochs":6,"resolution":"1024,1024","min_snr_gamma":5,"clip_skip":2,"save_every_n_epochs":1,"sample_every_n_epochs":1,"no_metadata":True}, "50枚前後の整備済みDatasetで、6epochまでの顔特徴の入りと過学習傾向を確認する。", "完成版とは限らない。epoch 4以降で過学習傾向が出る可能性があるため、必ずepoch別sampleを確認する。保存時メタデータ計算のメモリ負荷を避けるためno_metadataを使う。"),
    ("sdxl_2d_face_adamw8bit_standard", "SDXL 2D Face - AdamW8bit Standard", "SDXL", "sdxl_train_network.py", "SDXL向け2Dキャラクター顔LoRAの標準設定。", 20, 50, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":10,"max_train_epochs":10,"resolution":[1024,1024]}, "顔の特徴を安定して入れつつ固定化を抑える初期仮説。", "効きが弱い場合はTE SoftまたはStrongを比較する。"),
    ("sdxl_2d_face_adamw8bit_te_soft", "SDXL 2D Face - AdamW8bit TE Soft", "SDXL", "sdxl_train_network.py", "トリガー語と顔特徴の結びつきを少し強める設定。", 20, 50, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0.000005,"text_encoder_lr2":0.000005,"network_train_unet_only":False,"cache_text_encoder_outputs":False,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":10,"max_train_epochs":10,"resolution":[1024,1024]}, "Standardよりトリガー語への反応を少し高める。", "TE学習時はcache_text_encoder_outputsを使わない。"),
    ("sdxl_2d_face_adamw8bit_generalize", "SDXL 2D Face - AdamW8bit Generalize", "SDXL", "sdxl_train_network.py", "背景や構図の固定化を避ける弱めの設定。", 20, 60, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":16,"network_alpha":8,"train_batch_size":1,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "効きは弱めだが汎化を優先する。", "顔の再現が弱い場合はStandardへ戻す。"),
    ("sdxl_2d_face_adamw8bit_strong", "SDXL 2D Face - AdamW8bit Strong", "SDXL", "sdxl_train_network.py", "顔特徴が弱い場合の強め設定。", 20, 50, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0.00001,"text_encoder_lr2":0.00001,"network_train_unet_only":False,"cache_text_encoder_outputs":False,"network_dim":64,"network_alpha":32,"train_batch_size":1,"repeats":10,"max_train_epochs":12,"resolution":[1024,1024]}, "顔や髪型の特徴をより強く入れる。", "固定化しやすいので中盤epochも確認する。"),
    ("sdxl_2d_face_prodigy_test", "SDXL 2D Face - Prodigy Test", "SDXL", "sdxl_train_network.py", "Prodigyによる短期収束の比較試験。", 20, 50, {"optimizer_type":"Prodigy","lr_scheduler":"constant","learning_rate":1.0,"unet_lr":1.0,"text_encoder_lr1":0,"text_encoder_lr2":0,"optimizer_args":{"decouple":True,"weight_decay":0.01,"d_coef":1.0,"use_bias_correction":True},"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "AdamW8bitより早く強い特徴が出る可能性を見る。", "loss spikeと固定化に注意する。"),
    ("sdxl_2d_face_prodigy_soft", "SDXL 2D Face - Prodigy Soft", "SDXL", "sdxl_train_network.py", "Prodigy Testが強すぎる場合の抑制版。", 20, 60, {"optimizer_type":"Prodigy","lr_scheduler":"constant","learning_rate":1.0,"unet_lr":1.0,"text_encoder_lr1":0,"text_encoder_lr2":0,"optimizer_args":{"decouple":True,"weight_decay":0.01,"d_coef":0.5,"use_bias_correction":True},"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "Prodigyの収束性を使いつつ強すぎる学習を抑える。", "固定化する場合はAdamW8bitへ戻す。"),
    ("sd15_2d_face_adamw8bit_standard", "SD1.5 2D Face - AdamW8bit Standard", "SD1.5", "train_network.py", "SD1.5向け2Dキャラクター顔LoRAの標準設定。", 10, 40, {"mixed_precision":"fp16","save_precision":"fp16","optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr":0.00001,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":10,"max_train_epochs":10,"resolution":[768,768],"clip_skip":2,"sdpa":True}, "SDXLより短時間で顔特徴の入りを確認できる。", "最終用途がSDXLなら別途SDXL設定と比較する。"),
    ("sd15_2d_face_adamw8bit_generalize", "SD1.5 2D Face - AdamW8bit Generalize", "SD1.5", "train_network.py", "SD1.5で過学習と固定化を避ける設定。", 10, 50, {"mixed_precision":"fp16","save_precision":"fp16","optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr":0,"network_dim":16,"network_alpha":8,"train_batch_size":1,"repeats":8,"max_train_epochs":8,"resolution":[768,768],"clip_skip":2,"sdpa":True}, "効きは弱めだが構図変化への耐性を優先する。", "顔が弱い場合はStandardへ。"),
    ("sdxl_2d_face_small_dataset", "SDXL 2D Face - Small Dataset", "SDXL", "sdxl_train_network.py", "10から20枚程度の少数素材向け。", 10, 20, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":16,"network_alpha":8,"train_batch_size":1,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "少数素材で固定化を抑えつつ顔特徴を確認する。", "seed違い検証を推奨する。"),
    ("sdxl_2d_face_medium_dataset", "SDXL 2D Face - Medium Dataset", "SDXL", "sdxl_train_network.py", "40から80枚程度の素材で学習stepを抑える設定。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":1,"repeats":6,"max_train_epochs":10,"resolution":[1024,1024]}, "枚数が多いぶんrepeatを下げ、総stepを抑える。", "caption品質の影響が大きいのでタグ集計確認を推奨する。"),
]

TARGET_STEP_DEFAULTS = {
    "integration_smoke_sdxl": (1, 2, 5, 1, "Integration smokeは完走確認用の最小stepです。"),
    "sdxl_2d_face_pilot_3epoch": (100, 150, 300, 3, "Pilotは品質完成ではなく導線確認とepoch差の確認用です。"),
    "sdxl_2d_face_pilot_generalize_3epoch": (100, 150, 300, 3, "Pilotは弱め設定の短時間確認用です。"),
    "sdxl_2d_face_standard_6epoch": (1200, 2000, 3500, 6, "6epoch standardは短時間の本番寄り確認用です。"),
    "default": (2500, 5000, 8000, 6, "AdamW系の標準目安です。データセットやoptimizerに合わせて調整してください。"),
}


OPTIMIZER_DEFINITIONS = [
    {
        "id": "AdamW8bit",
        "name": "AdamW8bit",
        "lr_meaning": "normal_lr",
        "category": "stable",
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_checkpoint_count": 6,
        "note": "通常LR指定で扱いやすい標準optimizerです。",
    },
    {
        "id": "PagedAdamW8bit",
        "name": "PagedAdamW8bit",
        "lr_meaning": "normal_lr",
        "category": "stable/memory",
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_checkpoint_count": 6,
        "note": "AdamW8bitに近い目安で、メモリ節約寄りの選択肢です。",
    },
    {
        "id": "Adafactor",
        "name": "Adafactor",
        "lr_meaning": "relative_step/fixed_profile",
        "category": "memory/advanced",
        "target_steps_min": 2200,
        "target_steps_recommended": 4500,
        "target_steps_max": 7500,
        "target_checkpoint_count": 6,
        "note": "relative_step設定と固定LR設定で意味が変わるためAdvanced扱いです。",
    },
    {
        "id": "Lion",
        "name": "Lion",
        "lr_meaning": "normal_lr",
        "category": "experimental",
        "target_steps_min": 1800,
        "target_steps_recommended": 3500,
        "target_steps_max": 6500,
        "target_checkpoint_count": 6,
        "note": "効きが強めに出ることがある実験的optimizerです。",
    },
    {
        "id": "DAdaptAdam",
        "name": "DAdaptAdam",
        "lr_meaning": "auto_lr_multiplier",
        "category": "advanced",
        "target_steps_min": 2000,
        "target_steps_recommended": 4000,
        "target_steps_max": 7000,
        "target_checkpoint_count": 6,
        "note": "LRは自動調整の倍率として扱います。",
    },
    {
        "id": "DAdaptLion",
        "name": "DAdaptLion",
        "lr_meaning": "auto_lr_multiplier",
        "category": "experimental",
        "target_steps_min": 1800,
        "target_steps_recommended": 3500,
        "target_steps_max": 6500,
        "target_checkpoint_count": 6,
        "note": "自動LR系かつLion系の実験的optimizerです。",
    },
    {
        "id": "Prodigy",
        "name": "Prodigy",
        "lr_meaning": "auto_lr_multiplier",
        "category": "advanced",
        "target_steps_min": 1800,
        "target_steps_recommended": 3500,
        "target_steps_max": 6500,
        "target_checkpoint_count": 6,
        "note": "LRは自動調整の倍率です。短期収束と固定化に注意してください。",
    },
]

OPTIMIZER_PROFILES = [
    {
        "id": "adamw8bit_stable_face",
        "optimizer_definition_id": "AdamW8bit",
        "name": "AdamW8bit Stable Face",
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_checkpoint_count": 6,
        "note": "2D顔LoRA向けの安定プロファイルです。",
    },
    {
        "id": "adamw8bit_pilot",
        "optimizer_definition_id": "AdamW8bit",
        "name": "AdamW8bit Pilot",
        "target_steps_min": 100,
        "target_steps_recommended": 150,
        "target_steps_max": 300,
        "target_checkpoint_count": 3,
        "note": "導線確認とepoch差確認用の短時間プロファイルです。",
    },
    {
        "id": "prodigy_face",
        "optimizer_definition_id": "Prodigy",
        "name": "Prodigy Face",
        "target_steps_min": 1800,
        "target_steps_recommended": 3500,
        "target_steps_max": 6500,
        "target_checkpoint_count": 6,
        "note": "Prodigyの短期収束を使う顔LoRA向けプロファイルです。",
    },
]

TRAINING_RECIPES = [
    {
        "id": "integration_smoke",
        "optimizer_profile_id": "adamw8bit_pilot",
        "name": "Integration Smoke",
        "target_steps_min": 1,
        "target_steps_recommended": 2,
        "target_steps_max": 5,
        "target_checkpoint_count": 1,
        "note": "品質評価ではなく完走確認用です。",
    },
    {
        "id": "pilot_3epoch",
        "optimizer_profile_id": "adamw8bit_pilot",
        "name": "Pilot 3 Epoch",
        "target_steps_min": 100,
        "target_steps_recommended": 150,
        "target_steps_max": 300,
        "target_checkpoint_count": 3,
        "note": "品質完成前の短時間確認用です。",
    },
    {
        "id": "sdxl_face_standard",
        "optimizer_profile_id": "adamw8bit_stable_face",
        "name": "SDXL Face Standard",
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_checkpoint_count": 6,
        "note": "SDXL顔LoRAの標準Recipe目安です。",
    },
    {
        "id": "sdxl_face_standard_6epoch",
        "optimizer_profile_id": "adamw8bit_stable_face",
        "name": "SDXL Face Standard 6 Epoch",
        "target_steps_min": 1200,
        "target_steps_recommended": 2000,
        "target_steps_max": 3500,
        "target_checkpoint_count": 6,
        "note": "短時間の本番寄り確認用Recipeです。",
    },
    {
        "id": "sdxl_face_prodigy",
        "optimizer_profile_id": "prodigy_face",
        "name": "SDXL Face Prodigy",
        "target_steps_min": 1800,
        "target_steps_recommended": 3500,
        "target_steps_max": 6500,
        "target_checkpoint_count": 6,
        "note": "Prodigy比較用Recipeです。",
    },
]

PRESET_RECIPE_MAP = {
    "integration_smoke_sdxl": "integration_smoke",
    "sdxl_2d_face_pilot_3epoch": "pilot_3epoch",
    "sdxl_2d_face_pilot_generalize_3epoch": "pilot_3epoch",
    "sdxl_2d_face_standard_6epoch": "sdxl_face_standard_6epoch",
    "sdxl_2d_face_prodigy_test": "sdxl_face_prodigy",
    "sdxl_2d_face_prodigy_soft": "sdxl_face_prodigy",
}


def target_steps_for_preset(preset_id: str):
    return TARGET_STEP_DEFAULTS.get(preset_id, TARGET_STEP_DEFAULTS["default"])


PRESETS = []
for preset_id, name, family, script, purpose, count_min, count_max, params, behavior, risk in PRESET_DEFS:
    merged = {**COMMON_PARAMS, **params}
    target_min, target_recommended, target_max, target_checkpoint_count, step_target_note = target_steps_for_preset(preset_id)
    PRESETS.append({
        "id": preset_id,
        "name": name,
        "model_family": family,
        "training_script": script,
        "purpose": purpose,
        "recommended_dataset": {"image_count_min": count_min, "image_count_max": count_max},
        "params": merged,
        "expected_behavior": behavior,
        "risk_note": risk,
        "target_steps_min": target_min,
        "target_steps_recommended": target_recommended,
        "target_steps_max": target_max,
        "target_checkpoint_count": target_checkpoint_count,
        "step_target_note": step_target_note,
        "training_recipe_id": PRESET_RECIPE_MAP.get(preset_id, "sdxl_face_standard"),
    })


def preset_rows(now: str):
    for preset in PRESETS:
        yield (
            preset["id"], preset["name"], preset["model_family"], preset["training_script"], preset["purpose"],
            json.dumps(preset["params"], ensure_ascii=False, indent=2),
            json.dumps(preset["recommended_dataset"], ensure_ascii=False),
            preset["expected_behavior"], preset["risk_note"],
            "CODEX_LoRA_Helper_MVP_Instructions.md / kohya-ss sd-scripts v0.10.5",
            1, None,
            preset["target_steps_min"], preset["target_steps_recommended"], preset["target_steps_max"],
            preset["target_checkpoint_count"], preset["step_target_note"],
            preset["training_recipe_id"],
            now, now,
        )


def optimizer_definition_rows(now: str):
    for item in OPTIMIZER_DEFINITIONS:
        yield (
            item["id"], item["name"], item["lr_meaning"], item["category"],
            item["target_steps_min"], item["target_steps_recommended"], item["target_steps_max"],
            item["target_checkpoint_count"], item["note"], now, now,
        )


def optimizer_profile_rows(now: str):
    for item in OPTIMIZER_PROFILES:
        yield (
            item["id"], item["optimizer_definition_id"], item["name"],
            item["target_steps_min"], item["target_steps_recommended"], item["target_steps_max"],
            item["target_checkpoint_count"], item["note"], now, now,
        )


def training_recipe_rows(now: str):
    for item in TRAINING_RECIPES:
        yield (
            item["id"], item["optimizer_profile_id"], item["name"],
            item["target_steps_min"], item["target_steps_recommended"], item["target_steps_max"],
            item["target_checkpoint_count"], item["note"], now, now,
        )
