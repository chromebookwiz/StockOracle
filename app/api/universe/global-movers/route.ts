import { NextRequest, NextResponse } from "next/server";


export async function GET(request: NextRequest) {
  const limit = request.nextUrl.searchParams.get("limit") || "60";
  const response = await fetch(`${request.nextUrl.origin}/api/predictions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      universe: [],
      discoverGlobalMovers: true,
      globalMoversLimit: Number(limit),
      topK: Number(limit),
    }),
    cache: "no-store",
  });
  const json = await response.json();
  const ranking = Array.isArray(json?.ranking) ? json.ranking : [];
  const symbols = ranking
    .map((row: { symbol?: unknown }) => (typeof row.symbol === "string" ? row.symbol : ""))
    .filter(Boolean);
  const proxied = NextResponse.json(
    response.ok
      ? {
          symbols,
          count: symbols.length,
          source: "derived-from-global-ranking",
        }
      : json,
    { status: response.status },
  );
  proxied.headers.set("Cache-Control", "no-store");
  return proxied;
}