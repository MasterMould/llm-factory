# ui/tab_models.py — Model Manager: categorised catalogue, manual download, configure
import dataclasses
import subprocess
from pathlib import Path

import streamlit as st

from core.config import (
    BackendMode, MODEL_DIR, MODEL_CONFIG,
    MODEL_CATALOGUE, MODEL_CATEGORIES, MODEL_RUN_CONFIG_FIELDS, ModelRunConfig,
)
from core.auth   import audit_log
from core.models import ModelManager, ModelConfigManager
from core.inference import OllamaClient


# ═══════════════════════════════════════════════════════════════════════════════
# Download tab helpers
# ═══════════════════════════════════════════════════════════════════════════════

_VRAM_BUDGET = 16.0   # Arc A770 VRAM in GB

def _vram_bar(vram_gb: float) -> str:
    """Return a mini colour-coded VRAM bar string."""
    pct = min(vram_gb / _VRAM_BUDGET, 1.0)
    filled = int(pct * 10)
    bar = "█" * filled + "░" * (10 - filled)
    if pct <= 0.6:
        colour = "green"
    elif pct <= 0.9:
        colour = "orange"
    else:
        colour = "red"
    return f":{colour}[{bar}]  `{vram_gb} GB / {_VRAM_BUDGET} GB`"


def _badges(m: dict) -> str:
    parts = []
    if m.get("recommended"): parts.append("⭐ Recommended")
    if m.get("new"):         parts.append("🔥 New")
    if m.get("vision"):      parts.append("👁️ Vision")
    if m.get("jinja"):       parts.append("⚠️ Needs --jinja")
    return "  ".join(parts)


def _model_card(m: dict, dest: Path, key_prefix: str):
    """Render one model card inside an expander."""
    vram = float(m["vram"])
    fits = vram <= _VRAM_BUDGET
    status_icon = "🟢" if fits else "🟡"

    label = f"{status_icon} **{m['name']}**"
    badges = _badges(m)
    if badges:
        label += f"  —  {badges}"

    with st.expander(label, expanded=False):
        col_info, col_action = st.columns([3, 1])

        with col_info:
            st.markdown(f"**{m['desc']}**")
            if m.get("notes"):
                st.caption(m["notes"])
            st.markdown(_vram_bar(vram))

            # Tag pills
            tag_str = "  ".join(f"`{t}`" for t in m.get("tags", []))
            if tag_str:
                st.markdown(tag_str)

            st.caption(
                f"📁 `{m['file']}`  \n"
                f"🔗 [View on HuggingFace]({m['url'].split('/resolve')[0]})"
            )

        with col_action:
            if dest.exists():
                size_gb = dest.stat().st_size / 1e9
                st.success(f"✅ Downloaded\n{size_gb:.1f} GB")
                if st.button("▶️ Set active", key=f"act_{key_prefix}"):
                    ModelManager.set_active(str(dest))
                    # Auto-apply recommended config tweaks
                    cfg = ModelConfigManager.load(str(dest))
                    if m.get("jinja") and not cfg.jinja:
                        cfg.jinja = True
                        ModelConfigManager.save(str(dest), cfg)
                    st.success("Active!")
                    st.warning("Restart engine to load.")
                    st.rerun()
            else:
                if not fits:
                    st.warning(f"⚠️ {vram} GB  \nExceeds A770  \n(CPU offload OK)")
                else:
                    st.info(f"VRAM: {vram} GB  \nFits ✓")
                if st.button("⬇️ Download", type="primary", key=f"dl_{key_prefix}"):
                    progress = st.progress(0, text=f"Starting download of {m['file']}…")
                    try:
                        proc = subprocess.Popen(
                            ["wget", "--progress=dot:mega", "-O", str(dest), m["url"]],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        for line in proc.stdout:
                            if "%" in line:
                                try:
                                    pct = int(line.strip().split("%")[0].split()[-1])
                                    progress.progress(
                                        min(pct, 100) / 100,
                                        text=f"Downloading… {pct}%"
                                    )
                                except Exception:
                                    pass
                        proc.wait()
                        if proc.returncode == 0 and dest.exists():
                            progress.progress(1.0, text="Complete!")
                            st.success(f"Downloaded {dest.stat().st_size/1e9:.1f} GB")
                            audit_log(
                                st.session_state.get("username", "system"),
                                "MODEL_DOWNLOAD", m["file"], True,
                            )
                            if m.get("jinja"):
                                st.info(
                                    "ℹ️ This model requires **--jinja**. "
                                    "It has been auto-enabled in its run config."
                                )
                                cfg = ModelConfigManager.load(str(dest))
                                cfg.jinja = True
                                ModelConfigManager.save(str(dest), cfg)
                            st.rerun()
                        else:
                            dest.unlink(missing_ok=True)
                            st.error("Download failed — check network / disk space.")
                    except Exception as e:
                        dest.unlink(missing_ok=True)
                        st.error(f"Download error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Configure tab helpers  (unchanged from previous version)
# ═══════════════════════════════════════════════════════════════════════════════

def _field_widget(field: dict, current_val, key_prefix: str):
    w    = field["widget"]
    key  = f"{key_prefix}_{field['key']}"
    label = field["label"]
    help_ = field["help"]
    if w == "int_slider":
        return st.slider(label, min_value=field["min"], max_value=field["max"],
                         step=field["step"], value=int(current_val), help=help_, key=key)
    if w == "float_slider":
        return st.slider(label, min_value=float(field["min"]), max_value=float(field["max"]),
                         step=float(field["step"]), value=float(current_val), help=help_, key=key)
    if w == "checkbox":
        return st.checkbox(label, value=bool(current_val), help=help_, key=key)
    if w == "select":
        opts = field["options"]
        idx  = opts.index(current_val) if current_val in opts else 0
        return st.selectbox(label, opts, index=idx, help=help_, key=key)
    return current_val


def _config_form(model_path: str) -> None:
    cfg      = ModelConfigManager.load(model_path)
    cfg_dict = dataclasses.asdict(cfg)
    groups   = {}
    for f in MODEL_RUN_CONFIG_FIELDS:
        groups.setdefault(f["group"], []).append(f)

    st.markdown(
        f"**Configuring:** `{Path(model_path).name}`  \n"
        "Each setting maps directly to a `llama-server` / `llama-cli` flag.  \n"
        "Hover over any label for a plain-English explanation.",
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
            new_cfg = ModelRunConfig(**{k: new_vals.get(k, cfg_dict[k]) for k in cfg_dict})
            if ModelConfigManager.save(model_path, new_cfg):
                st.success("Saved.")
                audit_log(st.session_state.get("username", "system"),
                          "MODEL_CONFIG_SAVE", Path(model_path).name)
            else:
                st.error("Save failed.")

    with col_reset:
        if st.button("↩️ Reset to defaults", key=f"reset_cfg_{model_path}"):
            if ModelConfigManager.save(model_path, ModelRunConfig()):
                st.success("Reset.")
                st.rerun()

    with col_preview:
        show_preview = st.toggle("🖥️ Preview command", key=f"prev_{model_path}")

    if show_preview:
        st.divider()
        current_cfg = ModelRunConfig(**{k: new_vals.get(k, cfg_dict[k]) for k in cfg_dict})
        tab_cli, tab_srv = st.tabs(["llama-cli", "llama-server"])
        with tab_cli:
            st.code(ModelConfigManager.to_command_preview(current_cfg, model_path, "./llama-cli"),
                    language="bash")
        with tab_srv:
            st.code(
                ModelConfigManager.to_command_preview(current_cfg, model_path, "./llama-server")
                + f"\n  --port 8080 \\\n  --host 0.0.0.0 \\\n  --api-key local",
                language="bash",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Main tab
# ═══════════════════════════════════════════════════════════════════════════════

def tab_models():
    st.header("🤖 Model Manager")

    if st.session_state.backend != BackendMode.RUSTAIKIT.value:
        st.subheader("Ollama Models")
        for m in OllamaClient.models():
            st.write(m)
        st.divider()
        custom = st.text_input("Pull model:")
        if st.button("📥 Pull") and custom:
            with st.spinner(f"Pulling {custom}…"):
                ok, msg = OllamaClient.pull(custom)
            st.success(msg) if ok else st.error(msg)
        return

    active_path = ModelManager.get_active_path()
    active_name = Path(active_path).name if active_path else "none"
    st.info(f"🟢 **Active model:** `{active_name}`  &ensp; stored in: `{MODEL_CONFIG}`")

    mtabs = st.tabs([
        "📋 Installed",
        "⬇️ Download",
        "🔗 Manual Download",
        "⚙️ Configure",
        "🔄 Switch",
        "🗑️ Remove",
    ])

    # ── 0: Installed ──────────────────────────────────────────────────────────
    with mtabs[0]:
        installed = ModelManager.list_installed()
        if not installed:
            st.warning(f"No .gguf files found in `{MODEL_DIR}`")
            st.info("Use the **Download** tab to fetch a model, or **Manual Download** for a custom URL.")
        else:
            st.success(f"{len(installed)} model(s) in `{MODEL_DIR}`")
            for m in installed:
                marker = "🟢 **" + m["name"] + "**" if m["is_active"] else "⚪ " + m["name"]
                c1, c2, c3 = st.columns([6, 1, 1])
                c1.markdown(marker)
                c2.caption(m["size_human"])
                cfg_file = ModelConfigManager._cfg_path(m["path"])
                c3.caption("⚙️ saved" if cfg_file.exists() else "defaults")

    # ── 1: Download (catalogue) ───────────────────────────────────────────────
    with mtabs[1]:
        # Header + filters
        col_cat, col_vram, col_search = st.columns([2, 1, 2])
        with col_cat:
            categories = ["🔍 All categories"] + list(MODEL_CATEGORIES.keys())
            cat_filter = st.selectbox(
                "Category", categories,
                help="Filter the catalogue by use-case category.",
                key="dl_cat_filter",
            )
        with col_vram:
            vram_limit = st.slider(
                "Max VRAM (GB)", 1.0, 20.0, float(_VRAM_BUDGET), 0.5,
                help="Hide models that need more VRAM than this. A770 = 16 GB.",
                key="dl_vram_filter",
            )
        with col_search:
            name_filter = st.text_input(
                "Search models",
                placeholder="e.g. coder, deepseek, gemma…",
                label_visibility="visible",
                key="dl_name_filter",
            )

        # Count badges
        total = len(MODEL_CATALOGUE)
        new_count  = sum(1 for m in MODEL_CATALOGUE if m.get("new"))
        rec_count  = sum(1 for m in MODEL_CATALOGUE if m.get("recommended"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Models in catalogue", total)
        c2.metric("🔥 New this quarter", new_count)
        c3.metric("⭐ Recommended", rec_count)
        st.divider()

        # Filter models
        visible = [
            m for m in MODEL_CATALOGUE
            if (cat_filter == "🔍 All categories" or m["category"] == cat_filter)
            and float(m["vram"]) <= vram_limit
            and (not name_filter or name_filter.lower() in m["name"].lower()
                 or name_filter.lower() in m.get("desc", "").lower()
                 or any(name_filter.lower() in t.lower() for t in m.get("tags", [])))
        ]

        if not visible:
            st.info("No models match your current filters.")
        else:
            # Group by category
            shown_cats = {}
            for m in visible:
                shown_cats.setdefault(m["category"], []).append(m)

            for cat_name, models in shown_cats.items():
                cat_desc = MODEL_CATEGORIES.get(cat_name, "")
                st.subheader(cat_name)
                if cat_desc:
                    st.caption(cat_desc)

                for m in models:
                    dest = MODEL_DIR / m["file"]
                    _model_card(m, dest, m["file"][:20].replace(".", "_"))

                st.divider()

    # ── 2: Manual Download ────────────────────────────────────────────────────
    with mtabs[2]:
        st.markdown(
            "Download any GGUF from a direct URL — HuggingFace, a local server, "
            "or any other host. Useful for models not in the catalogue, custom quants, "
            "or private repos."
        )
        st.divider()

        # HuggingFace quick-fill helper
        with st.expander("💡 HuggingFace URL helper"):
            st.markdown(
                "HuggingFace direct download URLs follow this pattern:  \n"
                "```\nhttps://huggingface.co/{owner}/{repo}/resolve/main/{filename}.gguf\n```  \n"
                "**Examples:**  \n"
                "- `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` → file `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`  \n"
                "- `unsloth/DeepSeek-R1-GGUF` → file `DeepSeek-R1-UD-IQ1_S.gguf`  \n\n"
                "Go to the model page, click a file → ⋮ menu → **Copy download link**."
            )

        with st.form("manual_dl_form"):
            url_input = st.text_input(
                "Download URL",
                placeholder="https://huggingface.co/bartowski/…/resolve/main/…Q4_K_M.gguf",
                help="Direct link to the .gguf file. Must end in .gguf.",
            )

            # Auto-fill filename from URL
            auto_name = url_input.split("/")[-1].split("?")[0] if url_input else ""
            filename_input = st.text_input(
                "Save as filename",
                value=auto_name,
                placeholder="my-model-Q4_K_M.gguf",
                help="File will be saved to the model directory with this name.",
            )

            col_set, col_cfg = st.columns(2)
            with col_set:
                set_active = st.checkbox(
                    "Set as active model after download",
                    value=True,
                )
            with col_cfg:
                open_cfg = st.checkbox(
                    "Open Configure tab after download",
                    value=False,
                )

            submitted = st.form_submit_button("⬇️ Start Download", type="primary")

        if submitted:
            if not url_input or not url_input.startswith("http"):
                st.error("Please enter a valid URL starting with http:// or https://")
            elif not filename_input.endswith(".gguf"):
                st.error("Filename must end in .gguf")
            else:
                dest = MODEL_DIR / filename_input
                if dest.exists():
                    st.warning(f"`{filename_input}` already exists ({dest.stat().st_size/1e9:.1f} GB). Delete it first if you want to re-download.")
                else:
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    prog = st.progress(0, text=f"Connecting to {url_input[:60]}…")
                    try:
                        proc = subprocess.Popen(
                            ["wget", "--progress=dot:mega", "-O", str(dest), url_input],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        for line in proc.stdout:
                            if "%" in line:
                                try:
                                    pct = int(line.strip().split("%")[0].split()[-1])
                                    prog.progress(min(pct, 100) / 100, text=f"Downloading… {pct}%")
                                except Exception:
                                    pass
                        proc.wait()
                        if proc.returncode == 0 and dest.exists():
                            prog.progress(1.0, text="Complete!")
                            size = dest.stat().st_size / 1e9
                            st.success(f"✅ Downloaded `{filename_input}` ({size:.2f} GB)")
                            audit_log(
                                st.session_state.get("username", "system"),
                                "MODEL_MANUAL_DL", filename_input, True,
                            )
                            if set_active:
                                ModelManager.set_active(str(dest))
                                st.info("Set as active model. Restart the engine to load it.")
                            if open_cfg:
                                st.session_state["cfg_model_select"] = filename_input
                            st.rerun()
                        else:
                            dest.unlink(missing_ok=True)
                            st.error("Download failed — check the URL and your network connection.")
                    except FileNotFoundError:
                        dest.unlink(missing_ok=True)
                        st.error("`wget` not found. Install it: `sudo apt install wget`")
                    except Exception as e:
                        dest.unlink(missing_ok=True)
                        st.error(f"Download error: {e}")

        # Show recent manual downloads (files not in catalogue)
        catalogue_files = {m["file"] for m in MODEL_CATALOGUE}
        installed = ModelManager.list_installed()
        custom = [m for m in installed if m["name"] not in catalogue_files]
        if custom:
            st.divider()
            st.subheader("🗂️ Your custom / manual models")
            for m in custom:
                c1, c2, c3 = st.columns([5, 1, 1])
                marker = "🟢 **" + m["name"] + "**" if m["is_active"] else "⚪ " + m["name"]
                c1.markdown(marker)
                c2.caption(m["size_human"])
                with c3:
                    if not m["is_active"] and st.button("▶️", key=f"cust_act_{m['name']}",
                                                         help="Set active"):
                        ModelManager.set_active(m["path"])
                        st.rerun()

    # ── 3: Configure ─────────────────────────────────────────────────────────
    with mtabs[3]:
        installed = ModelManager.list_installed()
        if not installed:
            st.info("No installed models. Download one first.")
        else:
            opts       = {m["name"]: m["path"] for m in installed}
            default_ix = next((i for i, m in enumerate(installed) if m["is_active"]), 0)
            chosen_name = st.selectbox(
                "Select model to configure:",
                list(opts.keys()),
                index=default_ix,
                key="cfg_model_select",
            )
            chosen_path = opts[chosen_name]

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

    # ── 4: Switch ─────────────────────────────────────────────────────────────
    with mtabs[4]:
        installed = ModelManager.list_installed()
        if not installed:
            st.info("No installed models.")
        else:
            opts   = {m["name"]: m["path"] for m in installed}
            choice = st.selectbox("Switch to:", list(opts.keys()))
            chosen = opts[choice]

            cfg = ModelConfigManager.load(chosen)
            with st.expander("⚙️ Config that will be used when this model starts"):
                for f in MODEL_RUN_CONFIG_FIELDS:
                    val = getattr(cfg, f["key"])
                    st.markdown(
                        f"- `{f['key']}` = **{val}**  "
                        f"<small>{f['label'].split('(')[0].strip()}</small>",
                        unsafe_allow_html=True,
                    )

            if st.button("✅ Set active"):
                ModelManager.set_active(chosen)
                st.success(f"Active → `{choice}`")
                st.warning("Restart the engine (Stack tab) to load the new model.")
                audit_log(st.session_state.username, "MODEL_SWITCH", choice)

    # ── 5: Remove ─────────────────────────────────────────────────────────────
    with mtabs[5]:
        installed = ModelManager.list_installed()
        if not installed:
            st.info("No models to remove.")
        else:
            opts  = {f"{m['name']} ({m['size_human']})": m["name"] for m in installed}
            label = st.selectbox("Remove:", list(opts.keys()))
            fname = opts[label]
            is_active = any(m["is_active"] and m["name"] == fname for m in installed)
            if is_active:
                st.warning("⚠️ This is the active model — removing it will auto-select the next one.")

            col_del, col_also, _ = st.columns([1, 2, 4])
            with col_del:
                confirm = st.button("🗑️ Confirm delete", type="primary")
            with col_also:
                del_cfg = st.checkbox("Also delete saved config", value=True)

            if confirm:
                ok, msg = ModelManager.delete(fname)
                if ok and del_cfg:
                    from core.config import MODEL_CONFIGS_DIR
                    cfg_file = MODEL_CONFIGS_DIR / f"{Path(fname).stem}.json"
                    if cfg_file.exists():
                        cfg_file.unlink()
                        msg += " + config"
                st.success(msg) if ok else st.error(msg)
                audit_log(st.session_state.username, "MODEL_DELETE", fname, ok)
                if ok:
                    st.rerun()
