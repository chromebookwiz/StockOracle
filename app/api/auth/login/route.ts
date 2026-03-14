import { NextResponse } from "next/server";

import { createSessionCookie, operatorAuthConfigured, sessionCookieName, validateOperatorCredentials } from "../../../../lib/auth";


export async function POST(request: Request) {
  if (!operatorAuthConfigured()) {
    return NextResponse.json({ detail: "Operator auth is not configured." }, { status: 503 });
  }

  const payload = (await request.json()) as { username?: string; password?: string };
  const username = payload.username?.trim() || "";
  const password = payload.password || "";

  if (!validateOperatorCredentials(username, password)) {
    return NextResponse.json({ detail: "Invalid operator credentials." }, { status: 401 });
  }

  const sessionToken = await createSessionCookie(username);
  const response = NextResponse.json({ authenticated: true, username });
  response.cookies.set({
    name: sessionCookieName(),
    value: sessionToken,
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 12,
  });
  response.headers.set("Cache-Control", "no-store");
  return response;
}