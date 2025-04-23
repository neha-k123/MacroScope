import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Set up
start_date = "2024-12-01"
end_date = "2025-04-23"
ticker = "UUP"

# Download USD Index (UUP ETF)
df = yf.download(ticker, start=start_date, end=end_date)["Close"]

# Trump-related events (date: label)
trump_events = {
    "2025-01-20": "Inauguration Day",
    "2025-02-03": "Deferred Resignation Plan for Fed. Employees",
    "2025-02-20": "Attack on Fed Chair",
    "2025-02-28": "Trump–Zelenskyy Oval Office meeting",
    "2025-04-02": "Tariff Announcement",
    "2025-04-09": "90-day Tariff Pause"
}

# Assign colors to events
event_colors = {
    "Inauguration Day": "orange",
    "Deferred Resignation Plan for Fed. Employees": "green",
    "Attack on Fed Chair": "purple",
    "Trump–Zelenskyy Oval Office meeting":"gray",
    "Tariff Announcement": "red",
    "90-day Tariff Pause": "teal"
}

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
df.plot(ax=ax, label="USD Index (UUP)", color="blue", linewidth=2)

# Annotate with vertical lines
for date_str, label in trump_events.items():
    ax.axvline(pd.to_datetime(date_str), color=event_colors[label], linestyle="--", linewidth=2)
    # ax.text(pd.to_datetime(date_str), ax.get_ylim()[1], label, rotation=90, verticalalignment='bottom',
    #         fontsize=9, color=event_colors[label])

# Axis labels and title
ax.set_title("USD Index (UUP)", fontsize=14)
ax.set_ylabel("Price")
ax.set_xlabel("Date")

# Legend for Trump events
custom_lines = [plt.Line2D([0], [0], color=color, lw=2, linestyle="--") for color in event_colors.values()]
ax.legend(custom_lines, list(event_colors.keys()), title="Trump Events", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)

# Layout and save
plt.tight_layout()
# plt.show()
plt.savefig("./scripts/usd_with_trump_events.png", dpi=300, bbox_inches="tight")
plt.close()
