import { NextRequest, NextResponse } from "next/server";

import { requireSession } from "../../../../lib/auth";


export async function POST(request: NextRequest) {
  try {
    await requireSession();
  } catch (error) {
    return NextResponse.json({ detail: error instanceof Error ? error.message : "Authentication required." }, { status: 401 });
  }

  const payload = (await request.json()) as Record<string, unknown>;
  const executionToken = process.env.STOCKORACLE_EXECUTION_TOKEN || "";

  const response = await fetch(`${request.nextUrl.origin}/api/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      executionAuthToken: executionToken,
    }),
    cache: "no-store",
  });
  const json = await response.json();
  return NextResponse.json(json, { status: response.status });
}