"""
eda.py — Exploratory Data Analysis: візуалізація всіх 4 сегментів.

Публічний API:
    plot_price_dynamics(segs)       → None   (медіанна ціна по місяцях)
    plot_room_distribution(segs)    → None   (розподіл за кімнатністю)
    plot_room_share(segs)           → None   (частка кімнатності у %)
    plot_wall_house_type(segs)      → None   (матеріали стін та тип будинку)
    plot_floor_distribution(segs)   → None   (розподіл поверховості)
    plot_area_distribution(segs)    → None   (розподіл площі)
    plot_missing_values(segs)       → None   (аналіз пропущених значень)
    plot_feature_correlation(df)    → None   (Spearman-кореляція фічей з ціною)

segs — список кортежів (city, ctype, df), де df — вже завантажений сегмент.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .config import SEG_COLORS
from .data import best_price_col


def _setup_segs(data: pd.DataFrame) -> list:
    """Швидкий шлях: будує список (city, ctype, df) з сирого датасету."""
    segs = []
    for city in ['Львів', 'Київ']:
        for ctype in ['Оренда', 'Продаж']:
            sub = data[(data['city'] == city) & (data['contract_type'] == ctype)].copy()
            segs.append((city, ctype, sub))
    return segs


# ── Динаміка ціни ─────────────────────────────────────────────────────────────

def plot_price_dynamics(segs: list, save: str = 'price_dynamics_4seg.png') -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    fig.suptitle('Медіанна ціна за м² — динаміка по місяцях', fontsize=15, fontweight='bold')

    for i, (city, ctype, df_s) in enumerate(segs):
        ax  = axes.flat[i]
        col = best_price_col(ctype, df_s)
        if col is None:
            ax.text(0.5, 0.5, 'Немає даних', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{city} — {ctype}', fontsize=12, fontweight='bold')
            continue

        tmp = df_s.copy()
        tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
        tmp[col]    = pd.to_numeric(tmp[col], errors='coerce')
        monthly     = tmp.groupby('date')[col].median().dropna()

        ax.plot(monthly.index, monthly.values, color=SEG_COLORS[i], linewidth=2, marker='o', markersize=3)
        ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=SEG_COLORS[i])
        ax.set_title(f'{city} — {ctype}', fontsize=12, fontweight='bold')
        ax.set_ylabel('грн/м²' if 'uah' in col else 'USD/м²', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ── Розподіл за кімнатністю ───────────────────────────────────────────────────

def plot_room_distribution(segs: list, save: str = 'room_distribution_4seg.png') -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Розподіл оголошень за кількістю кімнат', fontsize=15, fontweight='bold')

    for i, (city, ctype, df_s) in enumerate(segs):
        ax = axes.flat[i]
        rc = df_s['room_count'].dropna().astype(int).value_counts().sort_index().head(8)
        bars = ax.bar(rc.index.astype(str), rc.values, color=SEG_COLORS[i], edgecolor='white')
        ax.set_title(f'{city} — {ctype}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Кімнат'); ax.set_ylabel('Оголошень')
        ax.grid(True, axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + h*0.01, f'{int(h):,}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ── Частка кімнатності (%) ────────────────────────────────────────────────────

def plot_room_share(segs: list, save: str = 'room_share_4seg.png') -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Структура ринку за кімнатністю (% від сегменту)', fontsize=15, fontweight='bold')

    for i, (city, ctype, df_s) in enumerate(segs):
        ax   = axes.flat[i]
        rc   = df_s['room_count'].dropna().astype(int).value_counts(normalize=True).sort_index().head(8) * 100
        bars = ax.barh(rc.index.astype(str), rc.values, color=SEG_COLORS[i], edgecolor='white')
        ax.set_title(f'{city} — {ctype}', fontsize=12, fontweight='bold')
        ax.set_xlabel('%'); ax.set_ylabel('Кімнат')
        ax.grid(True, axis='x', alpha=0.3)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f}%',
                    va='center', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ── Матеріали стін і тип будинку ─────────────────────────────────────────────

def plot_wall_house_type(segs: list, save: str = 'wall_house_4seg.png') -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Матеріал стін та тип будинку (4 сегменти)', fontsize=15, fontweight='bold')

    for i, (city, ctype, df_s) in enumerate(segs):
        for j, col in enumerate(['wall_type', 'house_type']):
            ax = axes[i // 2, i % 2 * 2 + j]
            if col not in df_s.columns:
                ax.set_visible(False)
                continue
            vc = df_s[col].value_counts().head(8)
            ax.barh(vc.index, vc.values, color=SEG_COLORS[i], edgecolor='white')
            ax.set_title(f'{city} — {ctype}: {col}', fontsize=10, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ── Розподіл поверховості ─────────────────────────────────────────────────────

def plot_floor_distribution(segs: list, save: str = 'floor_dist_4seg.png') -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Розподіл поверховості квартир', fontsize=15, fontweight='bold')

    for i, (city, ctype, df_s) in enumerate(segs):
        ax = axes.flat[i]
        floors = pd.to_numeric(df_s.get('floor'), errors='coerce').dropna()
        ax.hist(floors.clip(1, 30), bins=30, color=SEG_COLORS[i], edgecolor='white', alpha=0.85)
        ax.set_title(f'{city} — {ctype}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Поверх'); ax.set_ylabel('Кількість')
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ── Розподіл площі ────────────────────────────────────────────────────────────

def plot_area_distribution(segs: list, save: str = 'area_dist_4seg.png') -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Розподіл загальної площі квартир', fontsize=15, fontweight='bold')

    for i, (city, ctype, df_s) in enumerate(segs):
        ax   = axes.flat[i]
        area = pd.to_numeric(df_s.get('area_total'), errors='coerce').dropna()
        area = area[(area >= 10) & (area <= 300)]
        ax.hist(area, bins=50, color=SEG_COLORS[i], edgecolor='white', alpha=0.85)
        ax.set_title(f'{city} — {ctype}', fontsize=12, fontweight='bold')
        ax.set_xlabel('м²'); ax.set_ylabel('Кількість')
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ── Пропущені значення ────────────────────────────────────────────────────────

def plot_missing_values(segs: list, save: str = 'missing_4seg.png') -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Аналіз пропущених значень по сегментах (%)', fontsize=15, fontweight='bold')

    key_cols = [
        'area_total', 'area_living', 'area_kitchen', 'floor', 'floor_count',
        'room_count', 'built_year', 'ceiling_height', 'wall_type', 'house_type',
        'district', 'latitude', 'longitude', 'price', 'price_m2_usd', 'price_uah',
    ]

    for i, (city, ctype, df_s) in enumerate(segs):
        ax   = axes.flat[i]
        cols = [c for c in key_cols if c in df_s.columns]
        miss = df_s[cols].isna().mean() * 100
        miss = miss.sort_values(ascending=False)
        colors = ['#e74c3c' if v > 30 else '#f39c12' if v > 10 else '#2ecc71' for v in miss.values]
        ax.barh(miss.index, miss.values, color=colors)
        ax.set_title(f'{city} — {ctype}', fontsize=12, fontweight='bold')
        ax.set_xlabel('% пропущених')
        ax.axvline(10, color='orange', ls='--', alpha=0.6, lw=1)
        ax.axvline(30, color='red',    ls='--', alpha=0.6, lw=1)
        ax.grid(True, axis='x', alpha=0.3)
        for j, v in enumerate(miss.values):
            if v > 0.5:
                ax.text(v + 0.5, j, f'{v:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


# ── Кореляція фічей з ціною ───────────────────────────────────────────────────

def plot_feature_correlation(df_feat: pd.DataFrame, save: str = 'feature_correlation.png') -> None:
    """Spearman-кореляція числових фіч з ціною (|r| > 0.05)."""
    price_col = 'price_m2_usd' if 'price_m2_usd' in df_feat.columns else 'price_uah'
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != price_col]

    y = pd.to_numeric(df_feat[price_col], errors='coerce')
    rows = []
    for col in num_cols:
        x = pd.to_numeric(df_feat[col], errors='coerce')
        ok = x.notna() & y.notna()
        if ok.sum() < 50:
            continue
        r, p = spearmanr(x[ok], y[ok])
        rows.append({'feature': col, 'spearman_r': r, 'p_value': p})

    corr_df = pd.DataFrame(rows).sort_values('spearman_r', key=abs, ascending=False)
    corr_df = corr_df[corr_df['spearman_r'].abs() > 0.05]

    fig, ax = plt.subplots(figsize=(10, max(6, len(corr_df) * 0.3)))
    colors = ['#e74c3c' if r > 0 else '#3498db' for r in corr_df['spearman_r']]
    ax.barh(corr_df['feature'], corr_df['spearman_r'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title(f'Spearman-кореляція з {price_col}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Spearman r')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()
