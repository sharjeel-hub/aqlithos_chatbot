import aqlithosLogo from './assets/aqlithos-logo.png'
import { useEffect, useRef, useState } from 'react'
import {
  ArrowUpRight,
  Bot,
  ExternalLink,
  MessageSquarePlus,
  Send,
  ShieldCheck,
  UserRound,
} from 'lucide-react'

type Source = {
  source_number: number
  document_id: string
  title: string
  url: string
  distance: number
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  isError?: boolean
}

type ChatApiResponse = {
  answer?: string
  project_id?: string
  sources?: Source[]
  detail?: string
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'

const suggestedQuestions = [
  'What services does Aqlithos provide?',
  'Tell me about Aqlithos.',
  'What AI solutions does your team work on?',
  'How can I contact Aqlithos?',
]

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  async function handleSend(question: string) {
    const cleanQuestion = question.trim()

    if (!cleanQuestion || isLoading) return

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      content: cleanQuestion,
    }

    setMessages((current) => [...current, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: cleanQuestion }),
      })

      const data: ChatApiResponse = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'The request could not be completed.')
      }

      if (!data.answer) {
        throw new Error('The assistant returned an empty response.')
      }

      const assistantMessage: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        content: data.answer,
        sources: data.sources ?? [],
      }

      setMessages((current) => [...current, assistantMessage])
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        content:
          error instanceof Error
            ? error.message
            : 'Unable to connect to the Aqlithos assistant.',
        isError: true,
      }

      setMessages((current) => [...current, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  function startNewChat() {
    setMessages([])
    setInput('')
  }

  return (
    <div className="flex min-h-screen bg-[#f7f9fc] text-slate-800">
      {/* Sidebar */}
      <aside className="hidden w-[260px] shrink-0 flex-col border-r border-slate-200 bg-white px-5 py-6 lg:flex">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-white p-1.5 shadow-sm">
  <img
    src={aqlithosLogo}
    alt="Aqlithos"
    className="h-full w-full object-contain"
  />
</div>
          <div>
            <h1 className="text-lg font-bold tracking-wide text-[#102747]">
              AQLITHOS
            </h1>
            <p className="text-[10px] font-semibold tracking-[0.2em] text-slate-400">
              THINK. BUILD. INSPIRE.
            </p>
          </div>
        </div>

        <button
          onClick={startNewChat}
          className="mt-10 flex items-center gap-3 rounded-xl border border-slate-200 px-4 py-3 text-sm font-semibold text-slate-700 transition hover:border-blue-200 hover:bg-blue-50"
        >
          <MessageSquarePlus size={18} />
          New conversation
        </button>

        <div className="mt-10">
          <p className="mb-4 text-[11px] font-bold uppercase tracking-[0.16em] text-slate-400">
            Your assistant
          </p>
          <div className="flex items-center gap-3 rounded-xl bg-blue-50 px-3 py-3">
            <Bot size={19} className="text-blue-700" />
            <span className="text-sm font-medium text-[#102747]">
              Aqlithos AI Assistant
            </span>
          </div>
        </div>

        <div className="mt-auto rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
            <ShieldCheck size={17} className="text-emerald-600" />
            Knowledge-based AI
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Answers are generated using information retrieved from the
            Aqlithos knowledge base.
          </p>
        </div>
      </aside>

      {/* Main area */}
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[72px] shrink-0 items-center justify-between border-b border-slate-200 bg-white px-5 sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-19 w-49 shrink-0 items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-white p-1.5 shadow-sm lg:hidden">
  <img
    src={aqlithosLogo}
    alt="Aqlithos"
    className="h-full w-full object-contain"
  />
</div>
            <div>
              <h2 className="text-sm font-bold text-[#102747]">
                Aqlithos AI Assistant
              </h2>
              <p className="mt-0.5 text-xs text-slate-500">
                Your intelligent guide to Aqlithos
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-emerald-100 bg-emerald-50 px-3 py-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            <span className="text-xs font-medium text-emerald-700">
              AI Assistant
            </span>
          </div>
        </header>

        <section className="flex flex-1 flex-col overflow-y-auto px-4 py-8 sm:px-8">
          <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col">
            {messages.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center py-10">
                <div className="mb-6 flex h-20 w-100 items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-white p-2 shadow-lg shadow-blue-950/10">
  <img
    src={aqlithosLogo}
    alt="Aqlithos"
    className="h-full w-full object-contain"
  />
</div>

                <p className="mb-3 text-xs font-bold uppercase tracking-[0.22em] text-blue-700">
                  Welcome to Aqlithos
                </p>

                <h2 className="max-w-2xl text-center text-3xl font-bold tracking-tight text-[#102747] sm:text-4xl">
                  How can we help you today?
                </h2>

                <p className="mt-4 max-w-xl text-center text-sm leading-6 text-slate-500 sm:text-base">
                  Ask us about our company, services, AI capabilities, and
                  solutions. Our AI assistant is ready to help.
                </p>

                <div className="mt-10 grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
                  {suggestedQuestions.map((question) => (
                    <button
                      key={question}
                      onClick={() => void handleSend(question)}
                      className="group flex min-h-[76px] items-center justify-between gap-4 rounded-xl border border-slate-200 bg-white px-5 py-4 text-left text-sm font-medium text-slate-700 shadow-sm transition hover:-translate-y-0.5 hover:border-blue-200 hover:shadow-md"
                    >
                      <span>{question}</span>
                      <ArrowUpRight
                        size={17}
                        className="shrink-0 text-slate-400 transition group-hover:text-blue-700"
                      />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-8 pb-8">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 sm:gap-4 ${
                      message.role === 'user' ? 'flex-row-reverse' : ''
                    }`}
                  >
                    <div
                      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                        message.role === 'user'
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-[#102747] text-white'
                      }`}
                    >
                      {message.role === 'user' ? (
                        <UserRound size={18} />
                      ) : (
                        <Bot size={19} />
                      )}
                    </div>

                    <div
                      className={`min-w-0 max-w-[85%] sm:max-w-[78%] ${
                        message.role === 'user' ? 'text-right' : ''
                      }`}
                    >
                      <p className="mb-2 text-xs font-semibold text-slate-500">
                        {message.role === 'user' ? 'You' : 'Aqlithos Assistant'}
                      </p>

                      <div
                        className={`rounded-2xl px-4 py-3.5 text-left text-sm leading-7 sm:px-5 ${
                          message.role === 'user'
                            ? 'rounded-tr-sm bg-[#102747] text-white'
                            : message.isError
                              ? 'border border-red-200 bg-red-50 text-red-700'
                              : 'rounded-tl-sm border border-slate-200 bg-white text-slate-700 shadow-sm'
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      </div>

                      {message.role === 'assistant' &&
                        message.sources &&
                        message.sources.length > 0 && (
                          <div className="mt-3 rounded-xl border border-slate-200 bg-white p-4 text-left">
                            <p className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-500">
                              Sources
                            </p>
                            <div className="space-y-2">
                              {message.sources.map((source) => (
                                <a
                                  key={`${message.id}-${source.source_number}`}
                                  href={source.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="flex items-center justify-between gap-3 rounded-lg bg-slate-50 px-3 py-2.5 text-xs text-slate-700 transition hover:bg-blue-50 hover:text-blue-800"
                                >
                                  <span className="min-w-0">
                                    <span className="mr-2 font-semibold text-blue-700">
                                      {source.source_number}.
                                    </span>
                                    <span>{source.title}</span>
                                  </span>
                                  <ExternalLink
                                    size={14}
                                    className="shrink-0"
                                  />
                                </a>
                              ))}
                            </div>
                          </div>
                        )}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex gap-3 sm:gap-4">
                    <div className="mb-6 flex h-20 w-20 items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-white p-2 shadow-lg shadow-blue-950/10">
  <img
    src={aqlithosLogo}
    alt="Aqlithos"
    className="h-full w-full object-contain"
  />
</div>
                    <div>
                      <p className="mb-2 text-xs font-semibold text-slate-500">
                        Aqlithos Assistant
                      </p>
                      <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-5 py-4 shadow-sm">
                        <span className="h-2 w-2 animate-bounce rounded-full bg-blue-500" />
                        <span className="h-2 w-2 animate-bounce rounded-full bg-blue-500 [animation-delay:150ms]" />
                        <span className="h-2 w-2 animate-bounce rounded-full bg-blue-500 [animation-delay:300ms]" />
                        <span className="ml-2 text-xs text-slate-500">
                          Preparing your answer...
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </section>

        {/* Message input */}
        <footer className="shrink-0 border-t border-slate-200 bg-white px-4 py-5 sm:px-8">
          <form
            onSubmit={(event) => {
              event.preventDefault()
              void handleSend(input)
            }}
            className="mx-auto flex w-full max-w-4xl items-center gap-3 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition focus-within:border-blue-400 focus-within:ring-4 focus-within:ring-blue-100"
          >
            <input
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask anything about Aqlithos..."
              maxLength={2000}
              disabled={isLoading}
              className="min-w-0 flex-1 bg-transparent px-3 py-2 text-sm text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60"
            />

            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              aria-label="Send message"
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[#102747] text-white transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Send size={18} />
            </button>
          </form>

          <p className="mt-3 text-center text-[11px] text-slate-400">
            AI-generated responses may not always be accurate. Verify important
            information using the linked sources.
          </p>
        </footer>
      </main>
    </div>
  )
}

export default App
