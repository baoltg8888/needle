-- Pandoc filter: turn the Markdown chapters into LaTeX that numbers itself.
--  * "# Chapter 3 — Methodology" -> \chapter{Methodology}; "## 3.4 Foo" -> \section{Foo}
--  * an image followed by "**Figure 3.1.** caption" -> one figure with that caption
--  * "**Table 4.1.** caption" followed by a table -> the table's caption
--  * "Figure 4.5", "Table 4.1", "Section 3.4" in the text -> \ref so numbers stay right
--  * figure paths ../figures/x.png -> figures/x.pdf (vector)

local function stringify(x) return pandoc.utils.stringify(x) end

-- Set by build.py --telu: emit the Telkom University template's chapter commands.
TELU = os.getenv("THESIS_TELU") == "1"

-- Header numbering --------------------------------------------------------
local function strip_number(header)
  local text = stringify(header.content)
  local num, rest = text:match("^Chapter%s+(%d+)%s*[—%-]+%s*(.*)$")
  if not num then num, rest = text:match("^Appendix%s+(%u)%s*[—%-]+%s*(.*)$") end
  if not num then num, rest = text:match("^(%d+%.%d+[%.%d]*)%s+(.*)$") end
  if num then
    header.content = pandoc.Inlines(rest)
    header.identifier = "sec:" .. num
  end
  -- TelU template: chapters are "BAB III / METHODOLOGY" pages, appendices "LAMPIRAN A".
  if TELU and header.level == 1 and num then
    local title = pandoc.utils.stringify(header.content):upper()
    if num:match("^%d+$") then
      local romans = { "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X" }
      return pandoc.RawBlock("latex", string.format(
        "\\telubab{%s}{BAB %s}{%s}{sec:%s}", num, romans[tonumber(num)], title, num))
    end
    return pandoc.RawBlock("latex", string.format(
      "\\telubab{%d}{LAMPIRAN %s}{%s}{sec:%s}", num:byte() - 64, num, title, num))
  end
  return header
end

-- Caption paragraphs ------------------------------------------------------
-- Returns kind ("Figure"/"Table"), number, and caption inlines without the label.
local function caption_para(block)
  if block.t ~= "Para" or #block.content == 0 then return nil end
  local first = block.content[1]
  if first.t ~= "Strong" then return nil end
  local kind, num = stringify(first.content):match("^(%a+)%s+([%d%.A-Z]+%d)%.?$")
  if kind ~= "Figure" and kind ~= "Table" then return nil end
  local rest = pandoc.Inlines({})
  for i = 2, #block.content do rest:insert(block.content[i]) end
  while #rest > 0 and (rest[1].t == "Space" or rest[1].t == "SoftBreak") do rest:remove(1) end
  return kind, num, rest
end

local function image_of(block)
  if block.t == "Figure" then
    local img
    block:walk({ Image = function(i) img = i end })
    return img
  end
  if block.t == "Para" and #block.content == 1 and block.content[1].t == "Image" then
    return block.content[1]
  end
  return nil
end

local function pdf_path(src)
  return (src:gsub("^%.%./figures/", "figures/"):gsub("%.png$", ".pdf"))
end

local function blocks(list)
  local out = pandoc.Blocks({})
  local i = 1
  while i <= #list do
    local b = list[i]
    local img = image_of(b)
    local kind, num, cap = nil, nil, nil
    if list[i + 1] then kind, num, cap = caption_para(list[i + 1]) end
    if img and kind == "Figure" then
      local image = pandoc.Image(pandoc.Inlines({}), pdf_path(img.src))
      out:insert(pandoc.Figure(pandoc.Plain({ image }), { long = { pandoc.Plain(cap) } },
        { id = "fig:" .. num }))
      i = i + 2
    elseif img then
      out:insert(pandoc.Para({ pandoc.Image(img.caption, pdf_path(img.src)) }))
      i = i + 1
    else
      kind, num, cap = caption_para(b)
      if kind == "Table" and list[i + 1] and list[i + 1].t == "Table" then
        local tbl = list[i + 1]
        tbl.caption = { long = { pandoc.Plain(cap) } }
        tbl.identifier = "tbl:" .. num
        out:insert(tbl)
        i = i + 2
      else
        out:insert(b)
        i = i + 1
      end
    end
  end
  return out
end

-- Cross-references --------------------------------------------------------
local prefixes = { Figure = "fig:", Table = "tbl:", Section = "sec:", Chapter = "sec:" }

local function refs(inlines)
  local out = pandoc.Inlines({})
  local i = 1
  while i <= #inlines do
    local a, s, c = inlines[i], inlines[i + 1], inlines[i + 2]
    local lead, word = "", nil
    if a.t == "Str" then lead, word = a.text:match("^(%p*)(%a+)$") end
    local prefix = word and prefixes[word]
    if prefix and s and s.t == "Space" and c and c.t == "Str" then
      local num, tail = c.text:match("^(%d+%.%d+)(.*)$")
      if word == "Chapter" then num, tail = c.text:match("^(%d+)(.*)$") end
      if num and not tail:match("^[–%-]") and not tail:match("^%.%d") then
        if lead ~= "" then out:insert(pandoc.Str(lead)) end
        out:insert(pandoc.RawInline("latex", word .. "~\\ref{" .. prefix .. num .. "}"))
        if tail ~= "" then out:insert(pandoc.Str(tail)) end
        i = i + 3
        goto continue
      end
    end
    out:insert(a)
    i = i + 1
    ::continue::
  end
  return out
end

-- Korean (Hangul) text appears only inside quoted HotpotQA paragraphs. The default
-- fonts have no Hangul glyphs, so it is replaced with a marker in the PDF.
-- UTF-8 lead bytes EA–ED cover U+A000–U+DFFF, which includes all Hangul syllables.
local HANGUL = "[\234-\237][\128-\191][\128-\191]"

local function drop_hangul(text)
  local marked = text:gsub(HANGUL, "\1")
  marked = marked:gsub("\1[\1 ]*\1", "\1"):gsub("\1", "[Hangul]")
  return marked
end

local function hangul_code(block)
  block.text = drop_hangul(block.text)
  if block.t == "CodeBlock" then
    -- fvextra's Verbatim wraps long lines instead of running off the page.
    return pandoc.RawBlock("latex", "\\begin{Verbatim}\n" .. block.text .. "\n\\end{Verbatim}")
  end
  -- Inline code gets break points after / _ . - so long file paths do not run into
  -- the margin. Plain \texttt (not \path) so it also works inside captions.
  local escapes = {
    ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$",
    ["&"] = "\\&", ["#"] = "\\#", ["%"] = "\\%", ["_"] = "\\_", ["^"] = "\\^{}",
    ["~"] = "\\textasciitilde{}", ["["] = "{[}", ["]"] = "{]}",
  }
  local tex = block.text:gsub("[\\{}$&#%%_^~%[%]]", escapes)
  tex = tex:gsub("(\\_)", "%1\\allowbreak{}"):gsub("([/%.%-])", "%1\\allowbreak{}")
  return pandoc.RawInline("latex", "\\texttt{" .. tex .. "}")
end

local function hangul_str(str)
  str.text = drop_hangul(str.text)
  return str
end

return {
  { CodeBlock = hangul_code, Code = hangul_code, Str = hangul_str },
  { Header = strip_number },
  { Blocks = blocks },
  { Inlines = refs },
}
