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
    ("sdxl_2d_face_pilot_3epoch", "SDXL 2D Face - Pilot 3 Epoch", "SDXL", "sdxl_train_network.py", "実用LoRA評価前の短時間テスト。品質完成ではなく、loss推移・epoch差・sample比較の確認用。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"cosine","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":2,"repeats":2,"max_train_epochs":3,"resolution":"1024,1024","min_snr_gamma":5,"clip_skip":2,"save_every_n_epochs":1,"sample_every_n_epochs":1}, "50枚前後のデータセットで、各epochの差とloss推移を短時間で確認する。", "完成品質ではない。顔再現が弱い可能性があるが、まずは評価導線の確認を優先する。"),
    ("sdxl_2d_face_pilot_generalize_3epoch", "SDXL 2D Face - Pilot Generalize 3 Epoch", "SDXL", "sdxl_train_network.py", "固定化・過学習を避ける弱めの短時間テスト。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"cosine","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":16,"network_alpha":8,"train_batch_size":2,"repeats":2,"max_train_epochs":3,"resolution":"1024,1024","min_snr_gamma":5,"clip_skip":2,"save_every_n_epochs":1,"sample_every_n_epochs":1}, "50枚前後のデータセットで、弱め設定のepoch差とloss推移を短時間で確認する。", "完成品質ではない。固定化を避けるぶん顔再現が弱い可能性がある。"),
    ("sdxl_2d_face_standard_6epoch", "SDXL 2D Face - Standard 6 Epoch", "SDXL", "sdxl_train_network.py", "Dataset整備後の本番寄り短時間学習。顔特徴の入り、epoch別変化、過学習傾向を見る。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"cosine","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":2,"repeats":2,"max_train_epochs":6,"resolution":"1024,1024","min_snr_gamma":5,"clip_skip":2,"save_every_n_epochs":1,"sample_every_n_epochs":1,"no_metadata":True}, "50枚前後の整備済みDatasetで、6epochまでの顔特徴の入りと過学習傾向を確認する。", "完成版とは限らない。epoch 4以降で過学習傾向が出る可能性があるため、必ずepoch別sampleを確認する。保存時メタデータ計算のメモリ負荷を避けるためno_metadataを使う。"),
    ("sdxl_2d_face_adamw8bit_standard", "SDXL 2D Face - AdamW8bit Standard", "SDXL", "sdxl_train_network.py", "SDXL向け2Dキャラクター顔LoRAの標準設定。", 20, 50, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":2,"repeats":10,"max_train_epochs":10,"resolution":[1024,1024]}, "顔の特徴を安定して入れつつ固定化を抑える初期仮説。", "効きが弱い場合はTE SoftまたはStrongを比較する。"),
    ("sdxl_2d_face_adamw8bit_te_soft", "SDXL 2D Face - AdamW8bit TE Soft", "SDXL", "sdxl_train_network.py", "トリガー語と顔特徴の結びつきを少し強める設定。", 20, 50, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0.000005,"text_encoder_lr2":0.000005,"network_train_unet_only":False,"cache_text_encoder_outputs":False,"network_dim":32,"network_alpha":16,"train_batch_size":2,"repeats":10,"max_train_epochs":10,"resolution":[1024,1024]}, "Standardよりトリガー語への反応を少し高める。", "TE学習時はcache_text_encoder_outputsを使わない。"),
    ("sdxl_2d_face_adamw8bit_generalize", "SDXL 2D Face - AdamW8bit Generalize", "SDXL", "sdxl_train_network.py", "背景や構図の固定化を避ける弱めの設定。", 20, 60, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":16,"network_alpha":8,"train_batch_size":2,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "効きは弱めだが汎化を優先する。", "顔の再現が弱い場合はStandardへ戻す。"),
    ("sdxl_2d_face_adamw8bit_strong", "SDXL 2D Face - AdamW8bit Strong", "SDXL", "sdxl_train_network.py", "顔特徴が弱い場合の強め設定。", 20, 50, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0.00001,"text_encoder_lr2":0.00001,"network_train_unet_only":False,"cache_text_encoder_outputs":False,"network_dim":64,"network_alpha":32,"train_batch_size":2,"repeats":10,"max_train_epochs":12,"resolution":[1024,1024]}, "顔や髪型の特徴をより強く入れる。", "固定化しやすいので中盤epochも確認する。"),
    ("sdxl_2d_face_prodigy_test", "SDXL 2D Face - Prodigy Test", "SDXL", "sdxl_train_network.py", "Prodigyによる短期収束の比較試験。", 20, 50, {"optimizer_type":"Prodigy","lr_scheduler":"constant","learning_rate":1.0,"unet_lr":1.0,"text_encoder_lr1":0,"text_encoder_lr2":0,"optimizer_args":{"decouple":True,"weight_decay":0.01,"d_coef":1.0,"use_bias_correction":True},"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":2,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "AdamW8bitより早く強い特徴が出る可能性を見る。", "loss spikeと固定化に注意する。"),
    ("sdxl_2d_face_prodigy_soft", "SDXL 2D Face - Prodigy Soft", "SDXL", "sdxl_train_network.py", "Prodigy Testが強すぎる場合の抑制版。", 20, 60, {"optimizer_type":"Prodigy","lr_scheduler":"constant","learning_rate":1.0,"unet_lr":1.0,"text_encoder_lr1":0,"text_encoder_lr2":0,"optimizer_args":{"decouple":True,"weight_decay":0.01,"d_coef":0.5,"use_bias_correction":True},"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":2,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "Prodigyの収束性を使いつつ強すぎる学習を抑える。", "固定化する場合はAdamW8bitへ戻す。"),
    ("sd15_2d_face_adamw8bit_standard", "SD1.5 2D Face - AdamW8bit Standard", "SD1.5", "train_network.py", "SD1.5向け2Dキャラクター顔LoRAの標準設定。", 10, 40, {"mixed_precision":"fp16","save_precision":"fp16","optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr":0.00001,"network_dim":32,"network_alpha":16,"train_batch_size":4,"repeats":10,"max_train_epochs":10,"resolution":[768,768],"clip_skip":2,"sdpa":True}, "SDXLより短時間で顔特徴の入りを確認できる。", "最終用途がSDXLなら別途SDXL設定と比較する。"),
    ("sd15_2d_face_adamw8bit_generalize", "SD1.5 2D Face - AdamW8bit Generalize", "SD1.5", "train_network.py", "SD1.5で過学習と固定化を避ける設定。", 10, 50, {"mixed_precision":"fp16","save_precision":"fp16","optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr":0,"network_dim":16,"network_alpha":8,"train_batch_size":4,"repeats":8,"max_train_epochs":8,"resolution":[768,768],"clip_skip":2,"sdpa":True}, "効きは弱めだが構図変化への耐性を優先する。", "顔が弱い場合はStandardへ。"),
    ("sdxl_2d_face_small_dataset", "SDXL 2D Face - Small Dataset", "SDXL", "sdxl_train_network.py", "10から20枚程度の少数素材向け。", 10, 20, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.00005,"unet_lr":0.00005,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":16,"network_alpha":8,"train_batch_size":2,"repeats":8,"max_train_epochs":8,"resolution":[1024,1024]}, "少数素材で固定化を抑えつつ顔特徴を確認する。", "seed違い検証を推奨する。"),
    ("sdxl_2d_face_medium_dataset", "SDXL 2D Face - Medium Dataset", "SDXL", "sdxl_train_network.py", "40から80枚程度の素材で学習stepを抑える設定。", 40, 80, {"optimizer_type":"AdamW8bit","lr_scheduler":"constant","learning_rate":0.0001,"unet_lr":0.0001,"text_encoder_lr1":0,"text_encoder_lr2":0,"network_train_unet_only":True,"cache_text_encoder_outputs":True,"network_dim":32,"network_alpha":16,"train_batch_size":2,"repeats":6,"max_train_epochs":10,"resolution":[1024,1024]}, "枚数が多いぶんrepeatを下げ、総stepを抑える。", "caption品質の影響が大きいのでタグ集計確認を推奨する。"),
]

PRESETS = []
for preset_id, name, family, script, purpose, count_min, count_max, params, behavior, risk in PRESET_DEFS:
    merged = {**COMMON_PARAMS, **params}
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
    })


def preset_rows(now: str):
    for preset in PRESETS:
        yield (
            preset["id"], preset["name"], preset["model_family"], preset["training_script"], preset["purpose"],
            json.dumps(preset["params"], ensure_ascii=False, indent=2),
            json.dumps(preset["recommended_dataset"], ensure_ascii=False),
            preset["expected_behavior"], preset["risk_note"],
            "CODEX_LoRA_Helper_MVP_Instructions.md / kohya-ss sd-scripts v0.10.5",
            1, None, now, now,
        )
