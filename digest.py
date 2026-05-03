import os
import re
import logging
from datetime import datetime, timedelta

import pytz
import anthropic
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PDT = pytz.timezone("America/Los_Angeles")
TRIVIAL_PATTERN = re.compile(
    r"^(<:[^>]+:>\s*)+$"
    r"|^([\U0001F300-\U0001FFFF\s]+)$"
    r"|^\+1$|^:\+1:$|^:thumbsup:$"
)
TRIVIAL_SUBTYPES = {"channel_join", "channel_leave", "channel_topic", "bot_message", "channel_archive"}
MIN_WORD_COUNT = 4
USER_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)>")


def compute_window():
    now_local = datetime.now(PDT)
    tonight_10pm = now_local.replace(hour=22, minute=0, second=0, microsecond=0)
    oldest = tonight_10pm - timedelta(hours=24)
    return oldest.timestamp(), tonight_10pm.timestamp(), oldest, tonight_10pm


def make_permalink(workspace_url, channel_id, ts):
    ts_clean = ts.replace(".", "")
    return f"{workspace_url.rstrip('/')}/archives/{channel_id}/p{ts_clean}"


def make_channel_url(workspace_url, channel_id):
    return f"{workspace_url.rstrip('/')}/archives/{channel_id}"


def is_trivial(msg):
    if msg.get("subtype", "") in TRIVIAL_SUBTYPES:
        return True
    if msg.get("bot_id"):
        return True
    text = msg.get("text", "").strip()
    if not text:
        return True
    if TRIVIAL_PATTERN.match(text):
        return True
    if len(text.split()) < MIN_WORD_COUNT:
        return True
    return False


def get_user_name(client, user_id, cache):
    if user_id in cache:
        return cache[user_id]
    try:
        resp = client.users_info(user=user_id)
        profile = resp["user"].get("profile", {})
        name = profile.get("display_name") or profile.get("real_name") or user_id
        cache[user_id] = name
        return name
    except SlackApiError:
        cache[user_id] = user_id
        return user_id


def resolve_mentions(text, client, user_cache):
    def replace(match):
        return get_user_name(client, match.group(1), user_cache)
    return USER_MENTION_RE.sub(replace, text)


def get_channels(client):
    channels = []
    cursor = None
    while True:
        resp = client.conversations_list(
            types="public_channel",
            exclude_archived=True,
            limit=200,
            cursor=cursor,
        )
        for ch in resp["channels"]:
            channels.append({"id": ch["id"], "name": ch["name"]})
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return channels


def get_messages(client, channel_id, oldest_ts, latest_ts, workspace_url, user_cache):
    messages = []
    cursor = None
    try:
        while True:
            try:
                resp = client.conversations_history(
                    channel=channel_id,
                    oldest=str(oldest_ts),
                    latest=str(latest_ts),
                    limit=200,
                    cursor=cursor,
                )
            except SlackApiError as e:
                if e.response["error"] == "not_in_channel":
                    client.conversations_join(channel=channel_id)
                    resp = client.conversations_history(
                        channel=channel_id,
                        oldest=str(oldest_ts),
                        latest=str(latest_ts),
                        limit=200,
                        cursor=cursor,
                    )
                else:
                    raise
            for msg in resp["messages"]:
                if not is_trivial(msg):
                    sender = get_user_name(client, msg.get("user", ""), user_cache) if msg.get("user") else "unknown"
                    text = resolve_mentions(msg.get("text", "").strip(), client, user_cache)
                    messages.append({
                        "sender": sender,
                        "text": text,
                        "permalink": make_permalink(workspace_url, channel_id, msg["ts"]),
                    })
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        log.warning(f"Skipping channel {channel_id}: {e.response['error']}")
        return []
    return messages


def get_structured_digest(channels_data, date_str, workspace_url, anthropic_client):
    channel_blocks = []
    for ch in channels_data:
        channel_url = make_channel_url(workspace_url, ch["id"])
        msgs = "\n".join(
            f"  [{m['sender']}] [permalink={m['permalink']}] {m['text']}"
            for m in ch["messages"]
        )
        channel_blocks.append(
            f"CHANNEL name={ch['name']} channel_url={channel_url}\n{msgs}"
        )

    channel_payload = "\n\n".join(channel_blocks)

    system_prompt = (
        "You are a briefing writer for a busy executive who is not in the day-to-day of the company. "
        "They read this digest every evening to stay informed and decide what to follow up on. "
        "Output ONLY the structured format specified — no extra text, no commentary."
    )

    user_prompt = f"""Analyze the Slack messages below and return a structured digest in EXACTLY this format:

CHANNEL|channel-name|channel-url
BULLET|Summary sentence.|inception-permalink
BULLET|Another summary sentence if a separate topic exists.|inception-permalink

CHANNEL|channel-name2|channel-url2
BULLET|Summary sentence.|inception-permalink

---

FORMAT RULES:
- Each line is either CHANNEL or BULLET, pipe-separated, exactly 3 fields per line.
- Use the exact channel-url and permalink URLs from the data — do not change them.
- For each BULLET, use the permalink of the FIRST message that started that conversation or topic.
- Blank line between each channel block.
- No header line, no intro, no conclusion — only CHANNEL and BULLET lines.
- If zero channels have meaningful activity, output the single word: QUIET

CONTENT RULES:
- You are briefing an executive who does NOT read Slack daily. Assume they have zero context.
- Use real names from the message data — never say "a team member" or "someone."
- Name specific people, companies, products, and tools mentioned in the messages.
- Group related messages into a single topic. Do not write one bullet per message.
- Each bullet must fully capture the key point so it makes sense without clicking the link.
- Skip chitchat, reactions, check-ins, and scheduling noise.
- Aim for 1–3 bullets per channel.
- If a channel has nothing worth surfacing after filtering, omit it entirely.

RAW MESSAGE DATA:

{channel_payload}
"""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()

    if raw.strip() == "QUIET":
        return []

    channels = []
    current_channel = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        kind, a, b = parts
        if kind == "CHANNEL":
            current_channel = {"name": a, "url": b, "bullets": []}
            channels.append(current_channel)
        elif kind == "BULLET" and current_channel is not None:
            current_channel["bullets"].append({"text": a, "permalink": b})

    return channels


def build_blocks(structured, date_str):
    blocks = []

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*Daily Digest: {date_str}*"},
    })

    for ch in structured:
        list_items = []
        for bullet in ch["bullets"]:
            list_items.append({
                "type": "rich_text_section",
                "elements": [
                    {"type": "text", "text": f"{bullet['text']} - "},
                    {"type": "link", "url": bullet["permalink"], "text": "Link"},
                ],
            })

        blocks.append({
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [
                        {"type": "link", "url": ch["url"], "text": f"#{ch['name']}"},
                    ],
                },
                {
                    "type": "rich_text_list",
                    "style": "bullet",
                    "indent": 0,
                    "elements": list_items,
                },
            ],
        })

    return blocks


def main():
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    workspace_url = os.environ["SLACK_WORKSPACE_URL"]
    digest_channel_id = os.environ["SLACK_DIGEST_CHANNEL_ID"]

    slack = WebClient(token=slack_token)
    ai = anthropic.Anthropic(api_key=anthropic_key)
    user_cache = {}

    oldest_ts, latest_ts, oldest_dt, latest_dt = compute_window()
    date_str = f"{latest_dt.strftime('%A, %B')} {latest_dt.day}, {latest_dt.year}"
    log.info(f"Digest date: {date_str}")

    channels = get_channels(slack)
    log.info(f"Found {len(channels)} public channels")

    channels_data = []
    for ch in channels:
        msgs = get_messages(slack, ch["id"], oldest_ts, latest_ts, workspace_url, user_cache)
        if msgs:
            log.info(f"  #{ch['name']}: {len(msgs)} messages")
            channels_data.append({"id": ch["id"], "name": ch["name"], "messages": msgs})
        else:
            log.info(f"  #{ch['name']}: skipped (no real activity)")

    log.info(f"Channels with activity: {len(channels_data)}")

    if not channels_data:
        fallback = f"*Daily Digest: {date_str}*\n\n_Quiet day — no notable activity across all channels._"
        slack.chat_postMessage(channel=digest_channel_id, text=fallback, unfurl_links=False, unfurl_media=False)
        log.info("Done (quiet day).")
        return

    structured = get_structured_digest(channels_data, date_str, workspace_url, ai)

    if not structured:
        fallback = f"*Daily Digest: {date_str}*\n\n_Quiet day — no notable activity across all channels._"
        slack.chat_postMessage(channel=digest_channel_id, text=fallback, unfurl_links=False, unfurl_media=False)
        log.info("Done (quiet day).")
        return

    blocks = build_blocks(structured, date_str)
    fallback = f"Daily Digest: {date_str} — " + ", ".join(f"#{ch['name']}" for ch in structured)

    log.info("Posting digest...")
    slack.chat_postMessage(
        channel=digest_channel_id,
        text=fallback,
        blocks=blocks,
        unfurl_links=False,
        unfurl_media=False,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
