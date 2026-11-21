import type { AnswerEvent, Hit, Settings } from './api'

export type Stage = 'routing' | 'retrieving' | 'generating' | 'done' | 'error'

export interface Turn {
  id: number
  question: string
  settings: Settings
  stage: Stage
  route?: { specs: string[]; reason: string }
  hits?: Hit[]
  reranked?: boolean
  retrievalTime?: number
  answer?: AnswerEvent
  error?: string
  startedAt: number
  elapsed?: number
}
