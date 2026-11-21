// Turn "[TS 24.501 §A.1]" citations in the LLM answer into markdown links "#cite-<citation>",
// which the Answer component renders as clickable references to the Sources panel.
// Mirrors the tolerant parser in rag/generator.py: Unicode spaces, 【】 brackets, grouped citations.

// Emphasis markers are absorbed only when they wrap the citation on both sides (*[..]*, **[..]**),
// so '**bold text [TS ..]**' keeps its bold intact. \1 matches empty when group 1 is absent.
const BRACKET_RE = /(\*{1,2}|_)?[[【]([^\]】]*TS\s*2[34]\.501[^\]】]*)[\]】]\1/g
const ONE_CITE_RE =
  /TS\s*(2[34]\.501)\s*§?\s*((?:Annex\s+)?[A-Za-z]?\.?\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*(?:~\d+)?)/g

export const CITE_PREFIX = '#cite-'

export function linkCitations(text: string): string {
  const clean = text.replace(/[\u202f\u00a0\u2009]/g, ' ')
  return clean.replace(BRACKET_RE, (whole, _emph: string | undefined, inner: string) => {
    const links = [...inner.matchAll(ONE_CITE_RE)].map(([, spec, section]) => {
      const citation = `TS ${spec} §${section}`
      return `[${spec} §${section}](${CITE_PREFIX}${encodeURIComponent(citation)})`
    })
    return links.length ? links.join(' ') : whole
  })
}

export function citationFromHref(href: string | undefined): string | null {
  if (!href?.startsWith(CITE_PREFIX)) return null
  return decodeURIComponent(href.slice(CITE_PREFIX.length))
}
