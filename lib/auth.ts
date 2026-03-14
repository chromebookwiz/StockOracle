import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";
import { timingSafeEqual } from "node:crypto";


const SESSION_COOKIE_NAME = "stockoracle_session";
const SESSION_DURATION_SECONDS = 60 * 60 * 12;


type SessionPayload = {
  username: string;
};


function getSessionSecret(): Uint8Array {
  const secret = process.env.STOCKORACLE_SESSION_SECRET;
  if (!secret) {
    throw new Error("STOCKORACLE_SESSION_SECRET must be set when operator auth is enabled.");
  }
  return new TextEncoder().encode(secret);
}


export function operatorAuthConfigured(): boolean {
  return Boolean(process.env.STOCKORACLE_OPERATOR_PASSWORD);
}


export function operatorUsername(): string {
  return process.env.STOCKORACLE_OPERATOR_USERNAME || "operator";
}


function safeEqual(left: string, right: string): boolean {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  if (leftBuffer.length !== rightBuffer.length) {
    return false;
  }
  return timingSafeEqual(leftBuffer, rightBuffer);
}


export function validateOperatorCredentials(username: string, password: string): boolean {
  const expectedPassword = process.env.STOCKORACLE_OPERATOR_PASSWORD;
  if (!expectedPassword) {
    return false;
  }
  return safeEqual(username, operatorUsername()) && safeEqual(password, expectedPassword);
}


export async function createSessionCookie(username: string): Promise<string> {
  return new SignJWT({ username })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(`${SESSION_DURATION_SECONDS}s`)
    .sign(getSessionSecret());
}


export async function readSessionFromToken(token: string | undefined): Promise<SessionPayload | null> {
  if (!token) {
    return null;
  }
  try {
    const verified = await jwtVerify(token, getSessionSecret());
    const username = typeof verified.payload.username === "string" ? verified.payload.username : null;
    if (!username) {
      return null;
    }
    return { username };
  } catch {
    return null;
  }
}


export async function currentSession(): Promise<SessionPayload | null> {
  const token = (await cookies()).get(SESSION_COOKIE_NAME)?.value;
  return readSessionFromToken(token);
}


export async function requireSession(): Promise<SessionPayload> {
  const session = await currentSession();
  if (!session) {
    throw new Error("Authentication required.");
  }
  return session;
}


export function sessionCookieName(): string {
  return SESSION_COOKIE_NAME;
}


