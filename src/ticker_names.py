"""
Single source of truth mapping HK equity tickers -> short company names,
plus helpers to label tickers and pair strings for plots and tables.

Names are intentionally short (for readable axis labels / tables). Company
names for tickers added in the 2020 re-pull were sourced from public HKEX /
Bloomberg listings; verify against Bloomberg (LONG_COMP_NAME) if used in the
final report. Note: 990 HK = Deep Source Holdings (formerly Theme International)
does not look like an energy/materials name and may be a stray in the pull.

This is for graph purposes, to streamline the pipeline, this script wouldn't be needed.
Intention here is for visualisation for final report.
"""

TICKER_NAMES = {
    # ---- Tech (HSTCH) ----
    "700 HK Equity":  "Tencent",
    "9988 HK Equity": "Alibaba",
    "1211 HK Equity": "BYD",
    "1810 HK Equity": "Xiaomi",
    "300 HK Equity":  "Midea",
    "981 HK Equity":  "SMIC",
    "9999 HK Equity": "NetEase",
    "3690 HK Equity": "Meituan",
    "9618 HK Equity": "JD.com",
    "9888 HK Equity": "Baidu",
    "9961 HK Equity": "Trip.com",
    "6690 HK Equity": "Haier",
    "1024 HK Equity": "Kuaishou",
    "1347 HK Equity": "Hua Hong Semi",
    "2015 HK Equity": "Li Auto",
    "6618 HK Equity": "JD Health",
    "992 HK Equity":  "Lenovo",
    "9866 HK Equity": "NIO",
    "9868 HK Equity": "XPeng",
    "1698 HK Equity": "Tencent Music",
    "9660 HK Equity": "Horizon Robotics",
    "20 HK Equity":   "SenseTime",
    "9626 HK Equity": "Bilibili",
    "241 HK Equity":  "Alibaba Health",
    "9863 HK Equity": "Leapmotor",
    "2382 HK Equity": "Sunny Optical",
    "285 HK Equity":  "BYD Electronic",
    "780 HK Equity":  "Tongcheng Travel",
    "3888 HK Equity": "Kingsoft",
    "268 HK Equity":  "Kingdee",

    # ---- Energy / Materials (HSCIE + HSCIM) ----
    "857 HK Equity":  "PetroChina",
    "883 HK Equity":  "CNOOC",
    "1088 HK Equity": "China Shenhua",
    "2899 HK Equity": "Zijin Mining",
    "386 HK Equity":  "Sinopec",
    "3993 HK Equity": "CMOC",
    "2259 HK Equity": "Zijin Gold Intl",
    "2600 HK Equity": "Aluminum Corp of China",
    "1898 HK Equity": "China Coal",
    "1772 HK Equity": "Ganfeng Lithium",
    "1787 HK Equity": "Shandong Gold",
    "1171 HK Equity": "Yankuang Energy",
    "358 HK Equity":  "Jiangxi Copper",
    "1818 HK Equity": "Zhaojin Mining",
    "1208 HK Equity": "MMG",
    "2099 HK Equity": "China Gold Intl",
    "2883 HK Equity": "China Oilfield Services",
    "2788 HK Equity": "Chuangxin Industries",
    "3939 HK Equity": "Wanguo Gold",
    "3858 HK Equity": "Jiaxin Intl Resources",
    "1258 HK Equity": "China Nonferrous",
    "3668 HK Equity": "Yancoal Australia",
    "3330 HK Equity": "Lingbao Gold",
    "1164 HK Equity": "CGN Mining",
    "2689 HK Equity": "Nine Dragons Paper",
    "2386 HK Equity": "Sinopec Engineering",
    "2610 HK Equity": "Nanshan Aluminium",
    "3899 HK Equity": "CIMC Enric",
    "189 HK Equity":  "Dongyue Group",
    "1378 HK Equity": "China Hongqiao",
    "297 HK Equity":  "Sinofert",
    "467 HK Equity":  "United Energy",
    "639 HK Equity":  "Shougang Fushan",
    "826 HK Equity":  "Tiangong Intl",
    "934 HK Equity":  "Sinopec Kantons",
    "975 HK Equity":  "Mongolian Mining",
    "990 HK Equity":  "Deep Source (ex-Theme Intl)",
    "1277 HK Equity": "Kinetic Development",
    "1907 HK Equity": "China Risun",
    "1921 HK Equity": "Dalipal",
    "2314 HK Equity": "Lee & Man Paper",
}


def short_name(ticker):
    """'700 HK Equity' -> 'Tencent'. Falls back to the raw ticker if unmapped."""
    return TICKER_NAMES.get(ticker, ticker)


def label_ticker(ticker):
    """'700 HK Equity' -> '700 (Tencent)'. Falls back to the raw ticker."""
    name = TICKER_NAMES.get(ticker)
    if name is None:
        return ticker
    return f"{ticker.split()[0]} ({name})"


def label_pair(pair, sep=" vs ", names_only=True):
    """Relabel a pair string with company names.

    '1347 HK Equity vs 268 HK Equity' -> 'Hua Hong Semi vs Kingdee'
    Accepts both 'A vs B' and 'A-B' (Hua Hong Semi) vs 268 (Kingdee)'.
    """
    raw_sep = sep if sep in pair else ("-" if "-" in pair else sep)
    legs = [p.strip() for p in pair.split(raw_sep)]
    fmt = short_name if names_only else label_ticker
    return " vs ".join(fmt(leg) for leg in legs)
