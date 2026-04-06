# ui/tab_models.py — Model Manager tab with per-model run configuration form
import dataclasses
from pathlib import Path

import streamlit as st

from core.config import (
    BackendMode, MODEL_DIR, MODEL_CONFIG, MODEL_CATALOGUE,
    MODEL_RUN_CONFIG_FIELDS, ModelRunConfig,
)
from core.auth   import audit_log
from core.models import ModelManager, ModelConfigManager
from core.inference import OllamaClient


# ── helpers ───────────────────────────────────────────────────────────────────

def _field_widget(field: dict, current_val, key_prefix: str):
    """
    Render the appropriate Streamlit widget for a config field.
    Returns the new value.  Uses field["help"] as the help= tooltip.
    """
    w    = field["widget"]
    key  = f"{key_prefix}_{field['key']}"
    label = field["label"]
    help_ = field["help"]

    if w == "int_slider":
        return st.slider(label, min_value=field["min"], max_value=field["max"],
                         step=field["step"], value=int(current_val),
                         help=help_, key=key)
    if w == "float_slider":
        return st.slider(label, min_value=float(field["min"]),
                         max_value=float(field["max"]), step=float(field["step"]),
                         value=float(current_val), help=help_, key=key)
    if w == "checkbox":
        return st.checkbox(label, value=bool(current_val), help=help_, key=key)
    if w == "select":
        opts = field["options"]
        idx  = opts.index(current_val) if current_val in opts else 0
        return st.selectbox(label, opts, index=idx, help=help_, key=key)
    if w == "int_input":
        return st.number_input(label, value=int(current_val),
                               min_value=field.get("min", 0),
                               max_value=field.get("max", 99999),
                               help=help_, key=key)
    return current_val   # fallback — no widget change


def _config_form(model_path: str) -> None:
    """
    Render the full configuration form for a model.
    Loads saved config, shows every parameter with descriptions,
    and writes back on Save.
    """
    cfg      = ModelConfigManager.load(model_path)
    cfg_dict = dataclasses.asdict(cfg)
    groups   = {}
    for f in MODEL_RUN_CONFIG_FIELDS:
        groups.setdefault(f["group"], []).append(f)

    st.markdown(
        f"**Configuring:** `{Path(model_path).name}`  \n"
        "Each setting maps directly to a `llama-server` / `llama-cli` flag.  \n"
        "Hover over any label for a plain-English explanation.",
        unsafe_allow_html=False,
    )
    st.divider()

    new_vals: dict = {}
    for group_name, fields in groups.items():
        st.subheader(group_name)
        for f in fields:
            new_vals[f["key"]] = _field_widget(f, cfg_dict[f["key"]], model_path[:12])
        st.divider()

    col_save, col_reset, col_preview, _ = st.columns([1, 1, 1, 4])

    with col_save:
        if st.button("💾 Save config", type="primary", key=f"save_cfg_{model_path}"):
            new_cfg = ModelRunConfig(**{
                k: new_vals.get(k, cfg_dict[k]) for k in cfg_dict
            })
            if ModelConfigManager.save(model_path, new_cfg):
                st.success("Configuration saved.")
                audit_log(
                    st.session_state.get("username", "system"),
                    "MODEL_CONFIG_SAVE", Path(model_path).name,
                )
            else:
                st.error("Failed to save — check file permissions.")

    with col_reset:
        if st.button("↩️ Reset to defaults", key=f"reset_cfg_{model_path}"):
            if ModelConfigManager.save(model_path, ModelRunConfig()):
                st.success("Reset to defaults.")
                st.rerun()

    with col_preview:
        show_preview = st.toggle("🖥️ Preview command", key=f"prev_{model_path}")

    if show_preview:
        st.divider()
        current_cfg = ModelRunConfig(**{
            k: new_vals.get(k, cfg_dict[k]) for k in cfg_dict
        })
        tab_cli, tab_srv = st.tabs(["llama-cli", "llama-server"])
        with tab_cli:
            st.caption("Interactive / one-shot inference:")
            st.code(
                ModelConfigManager.to_command_preview(current_cfg, model_path, "./llama-cli"),
                language="bash",
            )
        with tab_srv:
            st.caption("OpenAI-compatible server (used by this app):")
            st.code(
                ModelConfigManager.to_command_preview(current_cfg, model_path, "./llama-server")
                + f"\n  --port 8080 \\\n  --host 0.0.0.0 \\\n  --api-key local",
                language="bash",
            )


# ── main tab ──────────────────────────────────────────────────────────────────

def tab_models():
    st.header("🤖 Model Manager")

    if st.session_state.backend != BackendMode.RUSTAIKIT.value:
        # ── Ollama mode ───────────────────────────────────────────────────────
        st.subheader("Ollama Models")
        for m in OllamaClient.models():
            c1, c2 = st.columns([5, 1])
            c1.write(m)
        st.divider()
        custom = st.text_input("Pull model:")
        if st.button("📥 Pull") and custom:
            with st.spinner(f"Pulling {custom}…"):
                ok, msg = OllamaClient.pull(custom)
            st.success(msg) if ok else st.error(msg)
        return

    # ── rust-ai-kit mode ──────────────────────────────────────────────────────
    active_path = ModelManager.get_active_path()
    active_name = Path(active_path).name if active_path else "none"
    st.info(f"🟢 **Active:** `{active_name}`  &ensp; config: `{MODEL_CONFIG}`")

    mtabs = st.tabs([
        "📋 Installed",
        "⚙️ Configure",
        "⬇️ Download",
        "🔄 Switch",
        "🗑️ Remove",
    ])

    # ── 0: Installed ──────────────────────────────────────────────────────────
    with mtabs[0]:
        installed = ModelManager.list_installed()
        if not installed:
            st.warning(f"No .gguf files in `{MODEL_DIR}`")
            st.info("Use the Download tab to fetch a model.")
        else:
            st.success(f"{len(installed)} model(s) in `{MODEL_DIR}`")
            for m in installed:
                marker = "🟢 **" + m["name"] + "**" if m["is_active"] else "⚪ " + m["name"]
                c1, c2, c3 = st.columns([6, 1, 1])
                c1.markdown(marker)
                c2.caption(m["size_human"])
                # Quick badge: show if a saved config exists for this model
                cfg_path = ModelConfigManager._cfg_path(m["path"])
                c3.caption("⚙️ saved" if cfg_path.exists() else "defaults")

    # ── 1: Configure ─────────────────────────────────────────────────────────
    with mtabs[1]:
        installed = ModelManager.list_installed()
        if not installed:
            st.info("No installed models. Use the Download tab first.")
        else:
            # Let the user pick which model to configure (defaults to active)
            opts       = {m["name"]: m["path"] for m in installed}
            default_ix = next(
                (i for i, m in enumerate(installed) if m["is_active"]), 0
            )
            chosen_name = st.selectbox(
                "Select model to configure:",
                list(opts.keys()),
                index=default_ix,
                key="cfg_model_select",
            )
            chosen_path = opts[chosen_name]

            # Visual diff: show how this config differs from defaults
            saved_cfg   = ModelConfigManager.load(chosen_path)
            default_cfg = ModelRunConfig()
            saved_dict  = dataclasses.asdict(saved_cfg)
            def_dict    = dataclasses.asdict(default_cfg)
            diffs = {k: (def_dict[k], saved_dict[k])
                     for k in saved_dict if saved_dict[k] != def_dict[k]}
            if diffs:
                with st.expander(f"🔍 {len(diffs)} setting(s) differ from defaults"):
                    for k, (dv, sv) in diffs.items():
                        st.markdown(f"- **{k}**: default `{dv}` → saved `{sv}`")

            st.divider()
            _config_form(chosen_path)

    # ── 2: Download ───────────────────────────────────────────────────────────
    with mtabs[2]:
        st.caption(
            f"All models verified on Intel Arc A770 (16 GB).  \n"
            f"Destination: `{MODEL_DIR}`  \n"
            "Source: HuggingFace (bartowski GGUF collection)"
        )
        idx = st.selectbox(
            "Select model:",
            range(len(MODEL_CATALOGUE)),
            format_func=lambda i: (
                f"{MODEL_CATALOGUE[i]['name']}  —  {MODEL_CATALOGUE[i]['vram']} GB VRAM"
            ),
        )
        m    = MODEL_CATALOGUE[idx]
        dest = MODEL_DIR / m["file"]

        # Info card
        col_info, col_vram = st.columns([3, 1])
        with col_info:
            st.markdown(
                f"**File:** `{m['file']}`  \n"
                f"**Description:** {m['desc']}"
            )
        with col_vram:
            st.metric("VRAM", f"{m['vram']} GB")

        if dest.exists():
            st.success(f"Already downloaded ({dest.stat().st_size / 1e9:.1f} GB)")
            if st.button("Set as active"):
                ModelManager.set_active(str(dest))
                st.success(f"Active → `{m['file']}`")
                st.warning("⚠️ Restart the engine (Stack tab) to load it.")
        else:
            if st.button("⬇️ Download", type="primary"):
                with st.spinner(f"Downloading {m['file']}… (this may take several minutes)"):
                    ok, msg = ModelManager.download(m["file"], m["url"])
                if ok:
                    st.success(msg)
                    audit_log(st.session_state.username, "MODEL_DOWNLOAD", m["file"], True)
                    new_path = str(MODEL_DIR / m["file"])
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Set as active now"):
                            ModelManager.set_active(new_path)
                            st.rerun()
                    with c2:
                        if st.button("⚙️ Configure now"):
                            ModelManager.set_active(new_path)
                            st.session_state["cfg_model_select"] = m["name"]
                            st.rerun()
                else:
                    st.error(msg)

    # ── 3: Switch ─────────────────────────────────────────────────────────────
    with mtabs[3]:
        installed = ModelManager.list_installed()
        if not installed:
            st.info("No installed models.")
        else:
            opts   = {m["name"]: m["path"] for m in installed}
            choice = st.selectbox("Switch to:", list(opts.keys()))
            chosen = opts[choice]

            # Show current config for the target model
            cfg = ModelConfigManager.load(chosen)
            with st.expander("⚙️ Config that will be used when this model starts"):
                for f in MODEL_RUN_CONFIG_FIELDS:
                    val = getattr(cfg, f["key"])
                    st.markdown(f"- `{f['key']}` = **{val}**  "
                                f"<small>{f['label'].split('(')[0].strip()}</small>",
                                unsafe_allow_html=True)

            if st.button("✅ Set active"):
                ModelManager.set_active(chosen)
                st.success(f"Active model → `{choice}`")
                st.warning(
                    "⚠️ Restart the engine (Stack tab) to load the new model."
                )
                audit_log(st.session_state.username, "MODEL_SWITCH", choice)

    # ── 4: Remove ─────────────────────────────────────────────────────────────
    with mtabs[4]:
        installed = ModelManager.list_installed()
        if not installed:
            st.info("No models to remove.")
        else:
            opts  = {f"{m['name']} ({m['size_human']})": m["name"] for m in installed}
            label = st.selectbox("Remove:", list(opts.keys()))
            fname = opts[label]
            is_active = any(m["is_active"] and m["name"] == fname for m in installed)
            if is_active:
                st.warning("⚠️ This is the active model. Removing it will auto-select the next one.")

            col_del, col_also, _ = st.columns([1, 2, 4])
            with col_del:
                confirm = st.button("🗑️ Confirm delete", type="primary")
            with col_also:
                del_cfg = st.checkbox("Also delete saved config", value=True)

            if confirm:
                ok, msg = ModelManager.delete(fname)
                if ok and del_cfg:
                    # Remove the associated config JSON if present
                    from core.config import MODEL_CONFIGS_DIR
                    cfg_stem = Path(fname).stem
                    cfg_file = MODEL_CONFIGS_DIR / f"{cfg_stem}.json"
                    if cfg_file.exists():
                        cfg_file.unlink()
                        msg += " + config"
                st.success(msg) if ok else st.error(msg)
                audit_log(st.session_state.username, "MODEL_DELETE", fname, ok)
                if ok:
                    st.rerun()
