import { NextRequest, NextResponse } from "next/server";

import { requireSession } from "../../../../lib/auth";


export async function GET(request: NextRequest) {
  try {
    await requireSession();
  } catch (error) {
    return NextResponse.json({ detail: error instanceof Error ? error.message : "Authentication required." }, { status: 401 });
  }

  const mode = request.nextUrl.searchParams.get("mode") || "paper";
  const token = process.env.STOCKORACLE_EXECUTION_TOKEN || "";
  const response = await fetch(`${request.nextUrl.origin}/api/positions?mode=${encodeURIComponent(mode)}&executionAuthToken=${encodeURIComponent(token)}`, {
    cache: "no-store",
  });
  const json = await response.json();
  return NextResponse.json(json, { status: response.status });
}