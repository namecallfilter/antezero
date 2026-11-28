"""
Update Script for BUMod Joker Variants
Fetches the latest commit and release tags from OceanRamen/BUMod,
extracts any new or modified Joker card sprites across commit history,
and saves them into balatro_mod/assets/bu_variants/ with an updated manifest.json.
"""

import os
import subprocess
import hashlib
import json
import shutil
import sys

import tempfile

REPO_DIR = os.path.join(tempfile.gettempdir(), "bumod_git_cache")
REPO_URL = "https://github.com/OceanRamen/BUMod.git"
DEST_DIR = os.path.join(os.path.dirname(__file__), "balatro_mod", "assets", "bu_variants")

def sync_repo():
    print(f"[1/4] Syncing BUMod repository (cached in {REPO_DIR})...")
    if not os.path.exists(REPO_DIR):
        print(f"Cloning bare repository from {REPO_URL}...")
        subprocess.check_call(["git", "clone", "--bare", REPO_URL, REPO_DIR])
    else:
        print("Fetching latest commits and tags...")
        subprocess.check_call(["git", "--git-dir=" + REPO_DIR, "fetch", "origin", "+refs/heads/*:refs/heads/*", "--tags"])

def analyze_and_extract():
    print(f"[2/4] Analyzing commits and extracting unique Joker textures...")
    os.makedirs(DEST_DIR, exist_ok=True)

    out = subprocess.check_output(["git", "--git-dir=" + REPO_DIR, "rev-list", "--reverse", "HEAD"]).decode("utf-8")
    commits = out.strip().split("\n")
    print(f"Total commits analyzed: {len(commits)}")

    all_joker_files = set()
    for c in commits:
        try:
            tree = subprocess.check_output(["git", "--git-dir=" + REPO_DIR, "ls-tree", "-r", "--name-only", c]).decode("utf-8")
            for line in tree.strip().split("\n"):
                if "single_jokers" in line and line.startswith("assets/2x/"):
                    all_joker_files.add(line)
        except Exception:
            pass

    manifest = {}
    total_variants = 0

    for f in sorted(list(all_joker_files)):
        unique_blobs = {}
        key = f.split("/")[-1].replace(".png", "")
        for c in commits:
            try:
                blob = subprocess.check_output(["git", "--git-dir=" + REPO_DIR, "show", f"{c}:{f}"])
                h = hashlib.md5(blob).hexdigest()
                if h not in unique_blobs:
                    unique_blobs[h] = (c[:7], blob)
            except Exception:
                pass

        manifest[key] = []
        versions = list(unique_blobs.values())
        for idx, (commit_sha, blob) in enumerate(versions):
            v_name = f"{key}_v{idx+1}_{commit_sha}.png"
            v_path = os.path.join(DEST_DIR, v_name)
            with open(v_path, "wb") as fp:
                fp.write(blob)
            manifest[key].append({
                "filename": v_name,
                "commit": commit_sha,
                "version_index": idx + 1
            })
            total_variants += 1
        print(f"  - {key}: {len(versions)} unique sprite versions")

    manifest_path = os.path.join(DEST_DIR, "manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"[3/4] Successfully saved {total_variants} sprite variants and manifest.json.")

def build_lua_manifest():
    print(f"[4/4] Generating Lua manifest (bu_variants_manifest.lua)...")
    manifest_path = os.path.join(DEST_DIR, "manifest.json")
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    lua_lines = [
        "-- Generated automatically by update_bumod_variants.py",
        "-- Contains historical BUMod Joker texture variants.",
        "local BU_VARIANTS = {",
    ]

    for key, variants in sorted(manifest.items()):
        lua_lines.append(f'    ["{key}"] = {{')
        for v in variants:
            lua_lines.append(f'        {{ filename = "{v["filename"]}", commit = "{v["commit"]}", version = {v["version_index"]} }},')
        lua_lines.append("    },")
    lua_lines.append("}")
    lua_lines.append("")
    lua_lines.append("return BU_VARIANTS")
    lua_lines.append("")

    lua_dest = os.path.join(os.path.dirname(__file__), "balatro_mod", "src", "bu_variants_manifest.lua")
    with open(lua_dest, "w") as fp:
        fp.write("\n".join(lua_lines))

    print(f"Generated {lua_dest} with {len(manifest)} keys.")

if __name__ == "__main__":
    sync_repo()
    analyze_and_extract()
    build_lua_manifest()
    print("Done! All BUMod historical variants are up to date.")
