import { NextRequest, NextResponse } from "next/server";

import { requireSession } from "../../../../lib/auth";


export async function GET(request: NextRequest) {
  try {
    await requireSession();
  } catch (error) {
    return NextResponse.json({ detail: error instanceof Error ? error.message : "Authentication required." }, { status: 401 });
  }

  if (!process.env.STOCKORACLE_EXECUTION_TOKEN) {
    return NextResponse.json({ detail: "Execution is not configured on the server." }, { status: 503 });
  }

  const mode = request.nextUrl.searchParams.get("mode") || "paper";
  const token = process.env.STOCKORACLE_EXECUTION_TOKEN || "";
  const response = await fetch(`${request.nextUrl.origin}/api/positions?mode=${encodeURIComponent(mode)}&executionAuthToken=${encodeURIComponent(token)}`, {
    cache: "no-store",
  });
  const json = await response.json();
  const proxied = NextResponse.json(json, { status: response.status });
  proxied.headers.set("Cache-Control", "no-store");
  return proxied;
}