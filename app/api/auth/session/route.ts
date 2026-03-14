import { NextResponse } from "next/server";

import { currentSession, operatorAuthConfigured } from "../../../../lib/auth";


export async function GET() {
  const session = await currentSession();
  const response = NextResponse.json({
    configured: operatorAuthConfigured(),
    authenticated: Boolean(session),
    username: session?.username || null,
  });
  response.headers.set("Cache-Control", "no-store");
  return response;
}