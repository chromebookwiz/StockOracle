import { NextResponse } from "next/server";

import { sessionCookieName } from "../../../../lib/auth";


export async function POST() {
  const response = NextResponse.json({ authenticated: false });
  response.cookies.set({
    name: sessionCookieName(),
    value: "",
    path: "/",
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
    maxAge: 0,
  });
  response.headers.set("Cache-Control", "no-store");
  return response;
}